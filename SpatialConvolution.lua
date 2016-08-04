local SpatialConvolution, parent =
    torch.class('cudnn.SpatialConvolution', 'nn.SpatialConvolution')
local ffi = require 'ffi'
local algo = require 'cudnn.algo'
local errcheck = cudnn.errcheck

function SpatialConvolution:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH, groups)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.groups = groups or 1
    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
    return self
end

function SpatialConvolution:createWeightDescriptors()
    assert(cudnn.typemap[torch.typename(self.weight)] or not self.weight, 'Only Cuda supported duh!')
    assert(cudnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Cuda supported duh!')
    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end
    -- create filterDescriptor for weight
    return cudnn.createDescriptors(1, 'struct cudnnFilterStruct*[?]',
                                   'cudnnCreateFilterDescriptor', 'cudnnDestroyFilterDescriptor')
end

-- if you change the configuration of the module manually, call this
function SpatialConvolution:resetWeightDescriptors(desc)
    -- for compatibility
    self.groups = self.groups or 1
    self.weightDesc = SpatialConvolution.createWeightDescriptors(self)
    desc = desc or torch.IntTensor({self.nOutputPlane/self.groups,
                                    self.nInputPlane/self.groups,
                                    self.kH, self.kW})
    errcheck('cudnnSetFilterNdDescriptor', self.weightDesc[0],
             cudnn.typemap[torch.typename(self.weight)], 'CUDNN_TENSOR_NCHW', self.nDim,
             desc:data());
end

function SpatialConvolution:fastest(mode)
    if mode == nil then mode = true end
    self.fastest_mode = mode
    self.iDesc = nil
    return self
end

function SpatialConvolution:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iDesc = nil
    return self
end

function SpatialConvolution:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function SpatialConvolution:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end


function SpatialConvolution:checkInputChanged(input)
    self.nDim = self.nDim or 4
    assert(input:dim() == self.nDim)
    assert(input:isContiguous())
    self.iSize = self.iSize or torch.LongStorage(self.nDim):fill(0)
    self.groups = self.groups or 1
    if not self.weightDesc then self:resetWeightDescriptors() end
    if not self.iDesc or not self.oDesc or input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] or (self.nDim==5 and input:size(5) ~= self.iSize[5]) then
       self.iSize = input:size()

       assert(self.nInputPlane == input:size(2), 'input has to contain: '
                 .. self.nInputPlane
                 .. ' feature maps, but received input of size: '
                 .. input:size(1) .. ' x ' .. input:size(2) ..
                 ' x ' .. input:size(3) .. ' x ' .. input:size(4))
       return true
    end
    return false
end

function SpatialConvolution:createIODescriptors(input)
   local batch = true
   if input:dim() == 3 then
      input = input:view(1, input:size(1), input:size(2), input:size(3))
      batch = false
   end
   if SpatialConvolution.checkInputChanged(self, input) then
        -- create input descriptor
        local input_slice = input:narrow(2,1,self.nInputPlane/self.groups)
        self.iDesc = cudnn.toDescriptor(input_slice)
        -- create conv descriptor
        self.convDesc = cudnn.createDescriptors(1, 'struct cudnnConvolutionStruct*[?]',
                                                'cudnnCreateConvolutionDescriptor', 'cudnnDestroyConvolutionDescriptor')
        self.padH, self.padW = self.padH or 0, self.padW or 0
        local pad = torch.IntTensor({self.padH, self.padW})
        local stride = torch.IntTensor({self.dH, self.dW})
        local upscale = torch.IntTensor({1,1})
        errcheck('cudnnSetConvolutionNdDescriptor', self.convDesc[0],
                 2, pad:data(),
                 stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
                 cudnn.configmap(torch.type(self.weight)));


        -- get output shape, resize output
        local oSize = torch.IntTensor(self.nDim)
        local oSizeD = oSize:data()
        errcheck('cudnnGetConvolutionNdForwardOutputDim',
                 self.convDesc[0], self.iDesc[0],
                 self.weightDesc[0], self.nDim, oSizeD)
        oSize[2] = oSize[2] * self.groups
        self.output:resize(oSize:long():storage())

        local output_slice = self.output:narrow(2,1,self.nOutputPlane/self.groups)
        -- create descriptor for output
        self.oDesc = cudnn.toDescriptor(output_slice)
        self.oDescForBias = cudnn.toDescriptor(self.output)

        algo.prepareHash(self, input_slice, output_slice)

        -- create offsets for groups
        local iH, iW = input:size(3), input:size(4)
        local kH, kW = self.kH, self.kW
        local oH, oW = oSize[3], oSize[4]
        self.input_offset = self.nInputPlane / self.groups * iH * iW
        self.output_offset = self.nOutputPlane / self.groups * oH * oW
        self.weight_offset = self.nInputPlane / self.groups * self.nOutputPlane / self.groups * kH * kW

        if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
        end

   end
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

function SpatialConvolution:makeContiguous(input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function SpatialConvolution:updateOutput(input)
    input = SpatialConvolution.makeContiguous(self, input)
    self:createIODescriptors(input)
    if not self.fwdAlgType then
       algo.setupForwardAlgorithm(self)
    end
    for g = 0, self.groups - 1 do
        errcheck('cudnnConvolutionForward', cudnn.getHandle(),
                 one:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.convDesc[0], self.fwdAlgType,
                 self.extraBuffer:data(), self.extraBuffer:nElement() * self.extraBuffer.elementSize(),
                 zero:data(),
                 self.oDesc[0], self.output:data() + g*self.output_offset);
    end

    -- add bias
    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 one:data(), self.biasDesc[0], self.bias:data(),
                 one:data(), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)
    input, gradOutput = SpatialConvolution.makeContiguous(self, input, gradOutput)
    assert(gradOutput:dim() == self.nDim-1 or gradOutput:dim() == self.nDim, 'gradOutput has to be nDim or nDim-1');
    self:createIODescriptors(input)
    if not self.bwdDataAlgType then
       algo.setupBackwardDataAlgorithm(self)
    end

    for g = 0,self.groups - 1 do
        errcheck('cudnnConvolutionBackwardData', cudnn.getHandle(),
                 one:data(),
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdDataAlgType,
                 self.extraBuffer:data(), self.extraBuffer:nElement() * self.extraBuffer.elementSize(),
                 zero:data(),
                 self.iDesc[0], self.gradInput:data() + g*self.input_offset);
    end
    return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or torch.FloatTensor(1):fill(1.0)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale
    input, gradOutput = SpatialConvolution.makeContiguous(self, input, gradOutput)
    assert(gradOutput:dim() == self.nDim-1 or gradOutput:dim() == self.nDim, 'gradOutput has to be nDim or nDim-1');
    self:createIODescriptors(input)

    if not self.bwdFilterAlgType then
       algo.setupBackwardFilterAlgorithm(self)
    end

    -- gradBias
    if self.bias then
        errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 one:data(),
                 self.biasDesc[0], self.gradBias:data())
    end

    for g = 0, self.groups - 1 do
        -- gradWeight
        errcheck('cudnnConvolutionBackwardFilter', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdFilterAlgType,
                 self.extraBuffer:data(), self.extraBuffer:nElement() * self.extraBuffer.elementSize(),
                 one:data(),
                 self.weightDesc[0], self.gradWeight:data() + g*self.weight_offset);
    end
end

function SpatialConvolution:clearDesc()
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
    self.scaleT = nil
end

function SpatialConvolution:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function SpatialConvolution:clearState()
   self:clearDesc()
   nn.utils.clear(self, 'extraBuffer', '_input', '_gradOutput')
   return nn.Module.clearState(self)
end

return SpatialConvolution
