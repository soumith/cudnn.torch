local SpatialConvolution, parent =
    torch.class('cudnn.SpatialConvolution', 'nn.SpatialConvolution')
local ffi = require 'ffi'
local find = require 'cudnn.find'
local errcheck = cudnn.errcheck
local checkedCall = find.checkedCall

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
end

-- if you change the configuration of the module manually, call this
function SpatialConvolution:resetWeightDescriptors(desc)
    -- for compatibility
    self.groups = self.groups or 1
    assert(cudnn.typemap[torch.typename(self.weight)], 'Only Cuda supported duh!')
    assert(cudnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Cuda supported duh!')

    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end

    self.weightDesc = cudnn.setFilterDescriptor(
       { dataType = cudnn.typemap[torch.typename(self.weight)],
         filterDimA = desc or
            {self.nOutputPlane/self.groups,
             self.nInputPlane/self.groups,
             self.kH, self.kW}
       }
    )

    return self
end

function SpatialConvolution:fastest(mode)
    if mode == nil then mode = true end
    if not self.fastest_mode or self.fastest_mode ~= mode then
       self.fastest_mode = mode
       self.iDesc = nil
    end
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
    assert(input:isContiguous(),
           "input to " .. torch.type(self) .. " needs to be contiguous, but is non-contiguous")
    if not self.iSize or self.iSize:size() ~= input:dim() then
       self.iSize = torch.LongStorage(input:dim()):fill(0)
    end
    self.groups = self.groups or 1
    if not self.weightDesc then self:resetWeightDescriptors() end
    if not self.weightDesc then error "Weights not assigned!" end

    if not self.iDesc or not self.oDesc or input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] or (input:dim()==5 and input:size(5) ~= self.iSize[5]) then
       self.iSize = input:size()
       assert(self.nInputPlane == input:size(2),
              'input has to contain: '
                 .. self.nInputPlane
                 .. ' feature maps, but received input of size: '
                 .. input:size(1) .. ' x ' .. input:size(2) .. ' x ' .. input:size(3)
                 .. (input:dim()>3 and ' x ' .. input:size(4) ..
                        (input:dim()==5 and ' x ' .. input:size(5) or '') or ''))
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
        self.padH, self.padW = self.padH or 0, self.padW or 0
        -- those needed to calculate hash
        self.pad = {self.padH, self.padW}
        self.stride = {self.dH, self.dW}

        self.convDescData = { padA = self.pad,
             filterStrideA = self.stride,
             upscaleA = {1,1},
             dataType = cudnn.configmap(torch.type(self.weight))
        }

        self.convDesc = cudnn.setConvolutionDescriptor(self.convDescData)

        -- get output shape, resize output
        local oSize = torch.IntTensor(4)
        errcheck('cudnnGetConvolutionNdForwardOutputDim',
                 self.convDesc[0], self.iDesc[0],
                 self.weightDesc[0], 4, oSize:data())
        oSize[2] = oSize[2] * self.groups
        self.output:resize(oSize:long():storage())
        self.oSize = self.output:size()

        local output_slice = self.output:narrow(2,1,self.nOutputPlane/self.groups)
        -- create descriptor for output
        self.oDesc = cudnn.toDescriptor(output_slice)
        self.oDescForBias = cudnn.toDescriptor(self.output)

        find:prepare(self, input_slice, output_slice)

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
   return self
end

local function makeContiguous(self, input, gradOutput)
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
    input = makeContiguous(self, input)
    self:createIODescriptors(input)
    local finder = find.get()
    local fwdAlgo = finder:forwardAlgorithm(self, { self.iDesc[0], self.input_slice, self.weightDesc[0],
                                                    self.weight, self.convDesc[0], self.oDesc[0], self.output_slice})
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    for g = 0, self.groups - 1 do
        checkedCall(self,'cudnnConvolutionForward', cudnn.getHandle(),
                    cudnn.scalar(input, 1),
                    self.iDesc[0], input:data() + g*self.input_offset,
                    self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                    self.convDesc[0], fwdAlgo,
                    extraBuffer, extraBufferSize,
                    cudnn.scalar(input, 0),
                    self.oDesc[0], self.output:data() + g*self.output_offset);
    end

    -- add bias
    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 cudnn.scalar(input, 1), self.biasDesc[0], self.bias:data(),
                 cudnn.scalar(input, 1), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)
    assert(gradOutput:dim() == input:dim()-1 or gradOutput:dim() == input:dim()
              or (gradOutput:dim()==5 and input:dim()==4), 'Wrong gradOutput dimensions');
    input, gradOutput = makeContiguous(self, input, gradOutput)
    self:createIODescriptors(input)
    local finder = find.get()
    local bwdDataAlgo = finder:backwardDataAlgorithm(self, { self.weightDesc[0], self.weight, self.oDesc[0],
                                                             self.output_slice, self.convDesc[0], self.iDesc[0], self.input_slice })
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    for g = 0,self.groups - 1 do
        checkedCall(self,'cudnnConvolutionBackwardData', cudnn.getHandle(),
                    cudnn.scalar(input, 1),
                    self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                    self.oDesc[0], gradOutput:data() + g*self.output_offset,
                    self.convDesc[0],
                    bwdDataAlgo,
                    extraBuffer, extraBufferSize,
                    cudnn.scalar(input, 0),
                    self.iDesc[0], self.gradInput:data() + g*self.input_offset)
    end
    return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or self.weight.new(1)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = torch.type(self.weight) == 'torch.CudaDoubleTensor'
       and self.scaleT:double() or self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale
    input, gradOutput = makeContiguous(self, input, gradOutput)
    self:createIODescriptors(input)
    local finder = find.get()
    local bwdFilterAlgo = finder:backwardFilterAlgorithm(self, { self.iDesc[0], self.input_slice, self.oDesc[0],
                                                               self.output_slice, self.convDesc[0], self.weightDesc[0], self.weight})

    -- gradBias
    if self.bias then
        errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 cudnn.scalar(input, 1),
                 self.biasDesc[0], self.gradBias:data())
    end

    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
    for g = 0, self.groups - 1 do
        -- gradWeight
       checkedCall(self,'cudnnConvolutionBackwardFilter', cudnn.getHandle(),
                   self.scaleT:data(),
                   self.iDesc[0], input:data() + g*self.input_offset,
                   self.oDesc[0], gradOutput:data() + g*self.output_offset,
                   self.convDesc[0],
                   bwdFilterAlgo,
                   extraBuffer, extraBufferSize,
                   cudnn.scalar(input, 1),
                   self.weightDesc[0], self.gradWeight:data() + g*self.weight_offset);
    end

    return self.gradOutput
end

function SpatialConvolution:clearDesc()
    self.weightDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.oSize = nil
    self.scaleT = nil
    return self
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
   nn.utils.clear(self, '_input', '_gradOutput', 'input_slice', 'output_slice')
   return nn.Module.clearState(self)
end
