local Dropout, parent = torch.class('cudnn.Dropout','nn.Dropout')

local errcheck = cudnn.errcheck
local ffi = require 'ffi'

local function getSize(f, desc)
   local size = ffi.new'size_t[1]'
   errcheck(f, desc, size)
   return tonumber(size[0])
end

function Dropout:createIODescriptors(input)
   assert(input:isContiguous(), 'Non-contiguous inputs not supported yet');
   if not self.inplace then
       self.output:resizeAs(input)
   end

   local nElem = input:nElement()
   self.nElem = self.nElem or nElem -- this goes to the second branch only once
   if self.iDesc and nElem == self.nElem then return end
   self.nElem = nElem
   self.iDesc = cudnn.toDescriptor(input:view(1,1,1,nElem))

   -- initialize RNG for dropouts lazily (per device)
   cudnn.dropout_rng_states = cudnn.dropout_rng_states or {}
   local dev = cutorch.getDevice()
   if not cudnn.dropout_rng_states[dev] then
      local states_size = getSize('cudnnDropoutGetStatesSize', cudnn.getHandle())
      cudnn.dropout_rng_states[dev] = torch.CudaByteTensor(states_size)
   end

   if not self.dropDesc then
      self.dropDesc = ffi.new('struct cudnnDropoutStruct*[1]')
      errcheck('cudnnCreateDropoutDescriptor', self.dropDesc)
      local reserves_size = getSize('cudnnDropoutGetReserveSpaceSize', self.iDesc[0])
      self.reserves = self.reserves or torch.CudaByteTensor()
      self.reserves = self.reserves:cudaByte():resize(reserves_size)
      local state = cudnn.dropout_rng_states[dev]
      errcheck('cudnnSetDropoutDescriptor', self.dropDesc[0],
         cudnn.getHandle(), self.p,
         state:data(), state:nElement(), torch.seed())

      local function destroyADesc(a)
         if (a[0]) then
            errcheck('cudnnDestroyDropoutDescriptor', a[0]);
            a[0] = nil
         end
      end
      ffi.gc(self.dropDesc, destroyADesc)
   end
end

function Dropout:updateOutput(input)
   assert(self.v2)
   if self.inplace then
      self.output:set(input)
   else
      self.output:resizeAs(input)
   end
   self:createIODescriptors(input)
   local train = self.p > 0 or self.train
   if train then
      errcheck('cudnnDropoutForward', cudnn.getHandle(),
         self.dropDesc[0], 
         self.iDesc[0], input:data(),
         self.iDesc[0], self.output:data(),
         self.reserves:data(),
         self.reserves:nElement())
   elseif not self.inplace then
      self.output:copy(input)
   end
   return self.output
end

function Dropout:updateGradInput(input, gradOutput)
   assert(self.train)
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(gradOutput)
   end
   if self.p > 0 then
      errcheck('cudnnDropoutBackward', cudnn.getHandle(),
         self.dropDesc[0],
         self.iDesc[0], gradOutput:data(),
         self.iDesc[0], self.gradInput:data(),
         self.reserves:data(),
         self.reserves:nElement())
   elseif not self.inplace then
      self.gradInput:copy(self.gradOutput)
   end
   return self.gradInput
end

