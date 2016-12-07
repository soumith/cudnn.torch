local VolumetricFullConvolution, parent
   = torch.class('cudnn.VolumetricFullConvolution', 'nn.VolumetricFullConvolution')
local ffi = require 'ffi'
local find = require 'cudnn.find'
local errcheck = cudnn.errcheck
local checkedCall = find.checkedCall

local Convolution = cudnn.SpatialConvolution

-- if you change the configuration of the module manually, call this
function VolumetricFullConvolution:resetWeightDescriptors()
   return Convolution.resetWeightDescriptors(
      self,
      {self.nInputPlane, self.nOutputPlane, self.kT, self.kH, self.kW}
   )
end

function VolumetricFullConvolution:fastest(mode)
   return Convolution.fastest(self, mode)
end


function VolumetricFullConvolution:setMode(fmode, bdmode, bwmode)
   return Convolution.setMode(self, fmode, bdmode, bwmode)
end

function VolumetricFullConvolution:resetMode()
   return Convolution.resetMode(self)
end


function VolumetricFullConvolution:createIODescriptors(input)
   local batch = true
   if input:dim() == 4 then
      input = input:view(1, input:size(1), input:size(2),
                         input:size(3), input:size(4))
      batch = false
   end
   assert(input:dim() == 5 and input:isContiguous());
   self.iSize = self.iSize or torch.LongStorage(5):fill(0)
   if Convolution.checkInputChanged(self, input) then
         -- create input descriptor
         local input_slice = input[{{},{1,self.nInputPlane},{},{}}]
         self.iDesc = cudnn.toDescriptor(input_slice)
         -- create conv descriptor
         self.pad = {self.padT, self.padH, self.padW}
         self.stride = {self.dT, self.dH, self.dW}
         self.convDescData = { padA = self.pad, filterStrideA = self.stride,
                               dataType = cudnn.configmap(torch.type(self.weight))}
         self.convDesc = cudnn.setConvolutionDescriptor(self.convDescData)

        -- get output shape, resize output
        local iwidth = input:size(5)
        local iheight = input:size(4)
        local idepth = input:size(3)
        local owidth = (iwidth - 1) * self.dW - 2*self.padW + self.kW + self.adjW
        local oheight = (iheight - 1) * self.dH - 2*self.padH + self.kH + self.adjH
        local odepth = (idepth - 1) * self.dT - 2*self.padT + self.kT + self.adjT
        local oSize = torch.IntTensor({input:size(1), self.nOutputPlane, odepth, oheight, owidth})
        self.output:resize(oSize:long():storage())

        -- create descriptor for output
        local output_slice = self.output[{{},{1,self.nOutputPlane},{},{}}]
        self.oDesc = cudnn.toDescriptor(output_slice)
        self.oDescForBias = cudnn.toDescriptor(
            self.output:view(self.output:size(1),
                             self.output:size(2),
                             self.output:size(3)*self.output:size(4),
                             self.output:size(5)))
        self.input_offset = 0
        self.output_offset = 0
	self.weight_offset = 0
        find:prepare(self, input_slice, output_slice)
        if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4),
                                           self.output:size(5))
        end
   end
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

function VolumetricFullConvolution:updateOutput(input)
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)
    local finder = find.get()
    -- Because SpatialFullConvolution is performing the adjoint of the forward
    -- convolution operator, we need to swap the forward and backward passes.


    local bwdDataAlgo = finder:backwardDataAlgorithm(self, {self.weightDesc[0], self.weight,
                                                            self.iDesc[0],self.input_slice,
                                                            self.convDesc[0], self.oDesc[0], self.output_slice})
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()

    checkedCall(self, 'cudnnConvolutionBackwardData', cudnn.getHandle(),
                cudnn.scalar(input, 1),
                self.weightDesc[0], self.weight:data(),
                self.iDesc[0], input:data(),
                self.convDesc[0], bwdDataAlgo,
                extraBuffer, extraBufferSize,
                cudnn.scalar(input, 0),
                self.oDesc[0], self.output:data())

    -- add bias
    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 cudnn.scalar(input, 1), self.biasDesc[0], self.bias:data(),
                 cudnn.scalar(input, 1), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function VolumetricFullConvolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)

    assert(gradOutput:dim() == 4 or gradOutput:dim() == 5, 'gradOutput has to be 4D or 5D');
    assert(gradOutput:isContiguous(), 'gradOutput has to be contiguous')
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)
    local finder = find.get()
    local fwdAlgo = finder:forwardAlgorithm(self, {self.oDesc[0], self.output_slice,
                                                   self.weightDesc[0], self.weight,
                                 self.convDesc[0], self.iDesc[0], self.input_slice})
    local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()

    checkedCall(self,'cudnnConvolutionForward', cudnn.getHandle(),
                cudnn.scalar(input, 1),
                self.oDesc[0], gradOutput:data(),
                self.weightDesc[0], self.weight:data(),
                self.convDesc[0],
                fwdAlgo,
                extraBuffer, extraBufferSize,
                cudnn.scalar(input, 0),
                self.iDesc[0], self.gradInput:data());
    return self.gradInput
end

function VolumetricFullConvolution:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or self.weight.new(1)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = torch.type(self.weight) == 'torch.CudaDoubleTensor'
       and self.scaleT:double() or self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale

   input, gradOutput = makeContiguous(self, input, gradOutput)
   assert(gradOutput:dim() == 4 or gradOutput:dim() == 5,
          'gradOutput has to be a 4D or 5D tensor');
   self:createIODescriptors(input)
   if not self.weightDesc then self:resetWeightDescriptors() end
   -- gradBias

   local finder = find.get()
   local bwdFilterAlgo = finder:backwardFilterAlgorithm(self, {self.oDesc[0], self.output_slice,
                                                                self.iDesc[0], self.input_slice,
                                                  self.convDesc[0], self.weightDesc[0], self.weight})
   errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
            self.scaleT:data(),
            self.oDescForBias[0], gradOutput:data(),
            cudnn.scalar(input, 1),
            self.biasDesc[0], self.gradBias:data());
   local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
   -- gradWeight
   checkedCall(self, 'cudnnConvolutionBackwardFilter', cudnn.getHandle(),
               self.scaleT:data(),
               self.oDesc[0], gradOutput:data(),
               self.iDesc[0], input:data(),
               self.convDesc[0],
               bwdFilterAlgo,
               extraBuffer, extraBufferSize,
               cudnn.scalar(input, 1),
               self.weightDesc[0], self.gradWeight:data());
end

function VolumetricFullConvolution:clearDesc()
   return Convolution.clearDesc(self)
end

function VolumetricFullConvolution:write(f)
   self:clearDesc()
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function VolumetricFullConvolution:clearState()
   self:clearDesc()
   nn.utils.clear(self, 'extraBuffer', '_input', '_gradOutput')
   return nn.Module.clearState(self)
end
