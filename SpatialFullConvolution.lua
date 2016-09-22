local SpatialFullConvolution, parent =
    torch.class('cudnn.SpatialFullConvolution', 'nn.SpatialFullConvolution')
local ffi = require 'ffi'
local find = require 'cudnn.find'
local errcheck = find.errcheck

local Convolution = cudnn.SpatialConvolution

function SpatialFullConvolution:resetWeightDescriptors()
   return Convolution.resetWeightDescriptors(self, torch.IntTensor({self.nInputPlane,
                                                                    self.nOutputPlane,
                                                                    self.kH, self.kW}))
end

function SpatialFullConvolution:fastest(mode)
   return Convolution.fastest(self, mode)
end

function SpatialFullConvolution:setMode(fmode, bdmode, bwmode)
   return Convolution.setMode(self, fmode, bdmode, bwmode)
end

function SpatialFullConvolution:resetMode()
   return Convolution.resetMode(self)
end

function SpatialFullConvolution:noBias()
   return Convolution.noBias(self)
end

function SpatialFullConvolution:createIODescriptors(input)
    local batch = true
    if input:dim() == 3 then
        input = input:view(1, input:size(1), input:size(2), input:size(3))
        batch = false
    end
    assert(input:dim() == 4 and input:isContiguous());
    self.iSize = self.iSize or torch.LongStorage(4):fill(0)

    if Convolution.checkInputChanged(self, input) then
        -- create input descriptor
        local input_slice = input[{{},{1,self.nInputPlane},{},{}}]
        self.iDesc = cudnn.toDescriptor(input_slice)

        -- create conv descriptor
        self.convDesc = cudnn.createDescriptors(1, 'struct cudnnConvolutionStruct*[?]',
                                                'cudnnCreateConvolutionDescriptor', 'cudnnDestroyConvolutionDescriptor')
        self.pad = torch.IntTensor({self.padH, self.padW})
        self.stride = torch.IntTensor({self.dH, self.dW})
        local upscale = torch.IntTensor({1,1})
        errcheck(self,'cudnnSetConvolutionNdDescriptor', self.convDesc[0],
                 2, self.pad:data(),
                 self.stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
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

        self.input_offset = 0
        self.output_offset = 0
        self.weight_offset = 0

        find:prepare(self, input_slice, output_slice)

        if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
        end
    end
end

function SpatialFullConvolution:updateOutput(input)
    self:createIODescriptors(input)
    local finder = find.get()
    self.bdmode = finder:backwardDataAlgorithm(self, {self.weightDesc[0], self.weight,
                                                         self.iDesc[0],self.input_slice,
                                                         self.convDesc[0], self.oDesc[0], self.output_slice})

    finder:setCalculatedWorkspaceSize(true)
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()

    -- Because SpatialFullConvolution is performing the adjoint of the forward
    -- convolution operator, we need to swap the forward and backward passes.
    errcheck(self,'cudnnConvolutionBackwardData', cudnn.getHandle(),
             cudnn.scalar(input, 1),
             self.weightDesc[0], self.weight:data(),
             self.iDesc[0], input:data(),
             self.convDesc[0], self.bdmode,
             extraBuffer, extraBufferSize,
             cudnn.scalar(input, 0),
             self.oDesc[0], self.output:data())

    -- add bias
    if self.bias then
        errcheck(self,'cudnnAddTensor', cudnn.getHandle(),
                 cudnn.scalar(input, 1), self.biasDesc[0], self.bias:data(),
                 cudnn.scalar(input, 1), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function SpatialFullConvolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)

    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    assert(gradOutput:isContiguous(), 'gradOutput has to be contiguous')
    self:createIODescriptors(input)
    local finder = find.get()
    self.fmode = finder:forwardAlgorithm(self, {self.oDesc[0], self.output_slice,
                                                self.weightDesc[0], self.weight,
                                                self.convDesc[0], self.iDesc[0], self.input_slice})
    finder:setCalculatedWorkspaceSize(true)
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    errcheck(self,'cudnnConvolutionForward', cudnn.getHandle(),
             cudnn.scalar(input, 1),
             self.oDesc[0], gradOutput:data(),
             self.weightDesc[0], self.weight:data(),
             self.convDesc[0],
             self.fmode,
             extraBuffer, extraBufferSize,
             cudnn.scalar(input, 0),
             self.iDesc[0], self.gradInput:data());
    return self.gradInput
end

function SpatialFullConvolution:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or self.weight.new(1)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = torch.type(self.weight) == 'torch.CudaDoubleTensor'
       and self.scaleT:double() or self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale

    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4,
           'gradOutput has to be 3D or 4D');
    assert(gradOutput:isContiguous(), 'gradOutput has to be contiguous')
    self:createIODescriptors(input)
    local finder = find.get()
    self.bmode = finder:backwardFilterAlgorithm(self, {self.oDesc[0], self.output_slice,
                                                       self.iDesc[0], self.input_slice,
                                                       self.convDesc[0], self.weightDesc[0], self.weight})
    -- gradBias
    if self.bias then
        errcheck(self,'cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 cudnn.scalar(input, 1),
                 self.biasDesc[0], self.gradBias:data())
    end
    finder:setCalculatedWorkspaceSize(true)
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    -- gradWeight
    errcheck(self,'cudnnConvolutionBackwardFilter', cudnn.getHandle(),
             self.scaleT:data(),
             self.oDesc[0], gradOutput:data(),
             self.iDesc[0], input:data(),
             self.convDesc[0],
             self.bmode,
             extraBuffer, extraBufferSize,
             cudnn.scalar(input, 1),
             self.weightDesc[0], self.gradWeight:data())
end

function SpatialFullConvolution:clearDesc()
   return Convolution.clearDesc(self)
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

function SpatialFullConvolution:read(file, version)
   parent.read(self, file)
   self.adjW = self.adjW or 0
   self.adjH = self.adjH or 0
end
