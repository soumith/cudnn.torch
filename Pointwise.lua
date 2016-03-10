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
       self.gradInput:resizeAs(input)
       self.output:resizeAs(input)
   end
   local nElem = input:nElement()
   self.nElem = self.nElem or nElem -- this goes to the second branch only once
   if self.iDesc and nElem == self.nElem then return end
   self.nElem = nElem
   self.iDesc = cudnn.toDescriptor(input:view(1,1,1,nElem))
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

function Pointwise:updateOutput(input)
   self:createIODescriptors(input)
   local activDesc = ffi.new('struct cudnnActivationStruct*[1]')
   errcheck('cudnnCreateActivationDescriptor', activDesc)
   errcheck('cudnnSetActivationDescriptor', activDesc[0], self.mode, 'CUDNN_PROPAGATE_NAN', 0.0);
   if self.inplace then self.output:set(input) end
   errcheck('cudnnActivationForward',
            cudnn.getHandle(), activDesc[0],
            one:data(),
            self.iDesc[0], input:data(),
            zero:data(),
            self.iDesc[0], self.output:data());

   local function destroyActivationDesc(d)
       errcheck('cudnnDestroyActivationDescriptor', d[0]);
   end
   ffi.gc(activDesc, destroyActivationDesc)

   return self.output
end

function Pointwise:updateGradInput(input, gradOutput)
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   self:createIODescriptors(input)
   local activDesc = ffi.new('struct cudnnActivationStruct*[1]')
   errcheck('cudnnCreateActivationDescriptor', activDesc)
   errcheck('cudnnSetActivationDescriptor', activDesc[0], self.mode, 'CUDNN_PROPAGATE_NAN', 0.0);

   if self.inplace then self.output:set(input); self.gradInput:set(gradOutput) end
   errcheck('cudnnActivationBackward',
            cudnn.getHandle(), activDesc[0],
            one:data(),
            self.iDesc[0], self.output:data(),
            self.iDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(),
            zero:data(),
            self.iDesc[0], self.gradInput:data());

   local function destroyActivationDesc(d)
       errcheck('cudnnDestroyActivationDescriptor', d[0]);
   end
   ffi.gc(activDesc, destroyActivationDesc)

   return self.gradInput
end

function Pointwise:clearDesc()
   self.iDesc = nil
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
   self._gradOutput = nil
   return parent.clearState(self)
end
