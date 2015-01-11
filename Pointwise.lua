local Pointwise, parent = torch.class('cudnn._Pointwise','nn.Module')
local errcheck = cudnn.errcheck

function Pointwise:__init(inplace)
   parent.__init(self)
   self.inplace = inplace or false
end

function Pointwise:createIODescriptors(input)
   assert(self.mode, 'mode is not set. (trying to use base class?)');
   assert(input:isContiguous(), 'Non-contiguous inputs not supported yet');
   local nElem = input:nElement()
   self.nElem = self.nElem or nElem -- this goes to the second branch only once
   if self.iDesc and nElem == self.nElem then return end

   self.nElem = nElem
   self.iDesc = cudnn.toDescriptor(input:view(1,1,1,nElem))
   if not self.inplace then
      self.gradInput:resizeAs(input)
      self.output:resizeAs(input)
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
