local Pointwise, parent = torch.class('cudnn._Pointwise','nn.Module')
local errcheck = cudnn.errcheck

function Pointwise:__init(inplace)
   parent.__init(self)
   self.inplace = inplace or false
   self.iSize = torch.LongStorage(4):fill(0)
end

function Pointwise:createIODescriptors(input)
   assert(self.mode, 'mode is not set. (trying to use base class?)');
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
      if not self.inplace then
         self.gradInput:resizeAs(input)
         self.output:resizeAs(input)
         if not batch then
            self.gradInput = self.gradInput:view(self.gradInput:size(2),
                                                 self.gradInput:size(3),
                                                 self.gradInput:size(4))
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
         end
      end
   end
end

function Pointwise:updateOutput(input)
   self:createIODescriptors(input)
   if self.inplace then self.output = input end
   errcheck('cudnnActivationForward',
            cudnn.handle[cutorch.getDevice()-1], self.mode,
            self.iDesc[0], input:data(),
            self.iDesc[0], self.output:data());
   return self.output
end

function Pointwise:updateGradInput(input, gradOutput)
   assert((gradOutput:dim() == 4 or gradOutput:dim() == 3));
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new():resizeAs(gradOutput)
      self._gradOutput:copy(gradOutput)
      gradOutput = self._gradOutput
   end
   self:createIODescriptors(input)
   if self.inplace then self.output = input; self.gradInput = gradOutput end
   errcheck('cudnnActivationBackward',
            cudnn.handle[cutorch.getDevice()-1], self.mode,
            self.iDesc[0], self.output:data(),
            self.iDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(),
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end
