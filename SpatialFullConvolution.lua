local SpatialFullConvolution, parent =
    torch.class('cudnn.SpatialFullConvolution', 'nn.SpatialFullConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck
local algo = require 'cudnn.algo'

local autotunerCache = {}
autotunerCache[1] = {} -- forward
autotunerCache[2] = {} -- backwardFilter
autotunerCache[3] = {} -- backwardData

local SpatialConvolution = cudnn.SpatialConvolution

-- if you change the configuration of the module manually, call this
function SpatialFullConvolution:resetWeightDescriptors()
    self.weightDesc = SpatialConvolution.createWeightDescriptors(self)
    local desc = torch.IntTensor({self.nInputPlane,
                                  self.nOutputPlane,
                                  self.kH, self.kW})
    errcheck('cudnnSetFilterNdDescriptor', self.weightDesc[0],
             cudnn.typemap[torch.typename(self.weight)], 'CUDNN_TENSOR_NCHW', 4,
             desc:data());
end

function SpatialFullConvolution:fastest(mode)
   return SpatialConvolution.fastest(self)
end

function SpatialFullConvolution:setMode(fmode, bdmode, bwmode)
   return SpatialConvolution.setMode(self, fmode, bdmode, bwmode)
end

function SpatialFullConvolution:resetMode()
   return SpatialConvolution.resetMode(self)
end

function SpatialFullConvolution:noBias()
   return SpatialConvolution.noBias(self)
end

function SpatialFullConvolution:createIODescriptors(input)
    local batch = true
    if input:dim() == 3 then
        input = input:view(1, input:size(1), input:size(2), input:size(3))
        batch = false
    end
    if SpatialConvolution.checkInputChanged(self, input) then
        -- create input descriptor
        local input_slice = input[{{},{1,self.nInputPlane},{},{}}]
        self.iDesc = cudnn.toDescriptor(input_slice)

        -- create conv descriptor
        self.convDesc = cudnn.createDescriptors(1, 'struct cudnnConvolutionStruct*[?]',
                                                'cudnnCreateConvolutionDescriptor', 'cudnnDestroyConvolutionDescriptor')
        local pad = torch.IntTensor({self.padH, self.padW})
        local stride = torch.IntTensor({self.dH, self.dW})
        local upscale = torch.IntTensor({1,1})
        errcheck('cudnnSetConvolutionNdDescriptor', self.convDesc[0],
                 2, pad:data(),
                 stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
                 cudnn.configmap(torch.type(self.weight)));

        -- get output shape, resize output
        local iwidth = input:size(4)
        local iheight = input:size(3)
        local owidth = (iwidth - 1) * self.dW - 2*self.padW + self.kW + self.adjW
        local oheight = (iheight - 1) * self.dH - 2*self.padH + self.kH + self.adjH
        local oSize = torch.IntTensor({input:size(1), self.nOutputPlane, oheight, owidth})
        self.output:resize(oSize:long():storage())

        -- create descriptor for output
        local output_slice = self.output[{{},{1,self.nOutputPlane},{},{}}]
        self.oDesc = cudnn.toDescriptor(output_slice)
        self.oDescForBias = cudnn.toDescriptor(self.output)

        algo.prepareHash(self, input_slice, output_slice)

    end
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

function SpatialFullConvolution:updateOutput(input)
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)
    if not self.bwdDataAlgType then
       algo.setupBackwardDataAlgorithm(self, {self.weightDesc[0], self.iDesc[0],
                                              self.convDesc[0], self.oDesc[0]})
    end

    -- Because SpatialFullConvolution is performing the adjoint of the forward
    -- convolution operator, we need to swap the forward and backward passes.
    errcheck('cudnnConvolutionBackwardData', cudnn.getHandle(),
             one:data(),
             self.weightDesc[0], self.weight:data(),
             self.iDesc[0], input:data(),
             self.convDesc[0], self.bwdDataAlgType,
             self.extraBuffer:data(), self.extraBuffer:nElement() * self.extraBuffer.elementSize(),
             zero:data(),
             self.oDesc[0], self.output:data())

    -- add bias
    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 one:data(), self.biasDesc[0], self.bias:data(),
                 one:data(), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function SpatialFullConvolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)

    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    assert(gradOutput:isContiguous(), 'gradOutput has to be contiguous')
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)
    if not self.fwdDataAlgType then
       algo.setupForwardAlgorithm(self, {self.oDesc[0], self.weightDesc[0],
                                         self.convDesc[0], self.iDesc[0]})
    end

    errcheck('cudnnConvolutionForward', cudnn.getHandle(),
             one:data(),
             self.oDesc[0], gradOutput:data(),
             self.weightDesc[0], self.weight:data(),
             self.convDesc[0],
             self.fwdAlgType,
             self.extraBuffer:data(), self.extraBuffer:nElement() * self.extraBuffer.elementSize(),
             zero:data(),
             self.iDesc[0], self.gradInput:data());
    return self.gradInput
end

function SpatialFullConvolution:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or torch.FloatTensor(1):fill(1.0)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale

    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    assert(gradOutput:isContiguous(), 'gradOutput has to be contiguous')
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)
    if not self.bwdFilterAlgType then
       algo.setupBackwardFilterAlgorithm(self, {self.oDesc[0], self.iDesc[0],
                                                self.convDesc[0], self.weightDesc[0]})
    end

    -- gradBias
    if self.bias then
        errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 one:data(),
                 self.biasDesc[0], self.gradBias:data())
    end

    -- gradWeight
    errcheck('cudnnConvolutionBackwardFilter', cudnn.getHandle(),
             self.scaleT:data(),
             self.oDesc[0], gradOutput:data(),
             self.iDesc[0], input:data(),
             self.convDesc[0],
             self.bwdFilterAlgType,
             self.extraBuffer:data(), self.extraBuffer:nElement() * self.extraBuffer.elementSize(),
             one:data(),
             self.weightDesc[0], self.gradWeight:data())
end

function SpatialFullConvolution:clearDesc()
    self.weightDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.algType = nil
    self.fwdAlgType = nil
    self.bwdDataAlgType = nil
    self.bwdFilterAlgType = nil
    self.extraBuffer = nil
end

function SpatialFullConvolution:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function SpatialFullConvolution:clearState()
   self:clearDesc()
   return nn.Module.clearState(self)
end
