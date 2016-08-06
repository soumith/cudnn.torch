local Pointwise, parent = torch.class('cudnn._Pointwise','nn.Module')

local errcheck = cudnn.errcheck
local ffi = require 'ffi'

function Pointwise:__init(inplace)
   parent.__init(self)
   self.inplace = inplace or false
end

function Pointwise:createIODescriptors(input)
   assert(self.mode, 'mode is not set. (trying to use base class?)');
   assert(input:isContiguous(), 'Non-contiguous inputs not supported yet');
   if not self.inplace then
       self.output:resizeAs(input)
   end

   if not self.activDesc then
      self.activDesc = ffi.new('struct cudnnActivationStruct*[1]')
      errcheck('cudnnCreateActivationDescriptor', self.activDesc)
      errcheck('cudnnSetActivationDescriptor', self.activDesc[0], self.mode, 'CUDNN_PROPAGATE_NAN', self.ceiling or 0.0);

      local function destroyADesc(a)
         if (a[0]) then
            errcheck('cudnnDestroyActivationDescriptor', a[0]);
            a[0] = nil
         end
      end
      ffi.gc(self.activDesc, destroyADesc)
   end

   local nElem = input:nElement()
   self.nElem = self.nElem or nElem -- this goes to the second branch only once
   if self.iDesc and nElem == self.nElem then return end
   self.nElem = nElem
   self.iDesc = cudnn.toDescriptor(input:view(1,1,1,nElem))

end

function Pointwise:updateOutput(input)
   self:createIODescriptors(input)
   if self.inplace then self.output:set(input) end
   errcheck('cudnnActivationForward',
            cudnn.getHandle(), self.activDesc[0],
            cudnn.scalar(input, 1),
            self.iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            self.iDesc[0], self.output:data());
   return self.output
end

function Pointwise:updateGradInput(input, gradOutput)
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   self:createIODescriptors(input)
   if self.inplace then
      self.output:set(input);
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(input)
   end
   errcheck('cudnnActivationBackward',
            cudnn.getHandle(), self.activDesc[0],
            cudnn.scalar(input, 1),
            self.iDesc[0], self.output:data(),
            self.iDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end

function Pointwise:clearDesc()
   self.iDesc = nil
   self.activDesc = nil
end

function Pointwise:write(f)
   self:clearDesc()
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function Pointwise:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_gradOutput')
   return parent.clearState(self)
end
