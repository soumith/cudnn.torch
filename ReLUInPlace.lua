local ReLU, parent = torch.class('cudnn.ReLUInPlace','nn.Module')
local errcheck = cudnn.errcheck

function ReLU:__init()
   parent.__init(self)
   self.iSize = torch.LongStorage(4):fill(0)
   self.mode = 'CUDNN_ACTIVATION_RELU'
end

function ReLU:createIODescriptors(input)
   local batch = true
   if input:dim() == 3 then
      input = input:view(1, input:size(1), input:size(2), input:size(3))
      batch = false
   end
   assert(input:dim() == 4 and input:isContiguous());
   if not self.iDesc or
      input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
   or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
      self.iSize = input:size()
      self.iDesc = cudnn.toDescriptor(input)
   end
end

function ReLU:updateOutput(input)
   self:createIODescriptors(input)
   self.output = input -- save memory, dont use a special state for this module
   errcheck('cudnnActivationForward',
            cudnn.handle[cutorch.getDevice()-1], self.mode,
            self.iDesc[0], input:data(),
            self.iDesc[0], self.output:data());
   return self.output
end

function ReLU:updateGradInput(input, gradOutput)
   assert((gradOutput:dim() == 4 or gradOutput:dim() == 3));
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new():resizeAs(gradOutput)
      self._gradOutput:copy(gradOutput)
      gradOutput = self._gradOutput
   end
   self:createIODescriptors(input)
   self.gradInput = gradOutput  -- save memory, dont use a special state for this module
   errcheck('cudnnActivationBackward',
            cudnn.handle[cutorch.getDevice()-1], self.mode,
            self.iDesc[0], self.output:data(),
            self.iDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(),
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end
