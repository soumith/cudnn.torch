local LRN, parent = torch.class('cudnn.SpatialCrossMapLRN', 'nn.Module')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function LRN:__init(size, alpha, beta, k)
   parent.__init(self)
   self.size = size or 5
   self.alpha = alpha or 1e-4
   self.beta = beta or 0.75
   self.k = k or 1.0
   assert(self.size >= 1 and self.size <= 16, "size has to be between 1 and 16")
   assert(self.k >= 1e-5, "k has to be greater than 1e-5")
   assert(self.beta >= 0.01, "Beta has to be > 0.01")
end

function LRN:resetDescriptors()
   -- create LRN descriptor
   self.LRNDesc = ffi.new('struct cudnnLRNStruct*[1]')
   errcheck('cudnnCreateLRNDescriptor', self.LRNDesc)
   errcheck('cudnnSetLRNDescriptor', self.LRNDesc[0], self.size,
            self.alpha, self.beta, self.k);
   local function destroyDesc(d)
      errcheck('cudnnDestroyLRNDescriptor', d[0]);
   end
   ffi.gc(self.LRNDesc, destroyDesc)
end

function LRN:createIODescriptors(input)
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
      self.output:resizeAs(input)

      -- create input/output descriptor
      self.iDesc = cudnn.toDescriptor(input)
      if not batch then
         self.output = self.output:view(self.output:size(2),
                                        self.output:size(3),
                                        self.output:size(4))
      end
   end
end




function LRN:updateOutput(input)
   if self.K then self.k, self.K = self.K, nil end
   if not self.LRNDesc then self:resetDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnLRNCrossChannelForward', cudnn.getHandle(),
            self.LRNDesc[0],
            'CUDNN_LRN_CROSS_CHANNEL_DIM1',
            cudnn.scalar(input, 1),
            self.iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            self.iDesc[0], self.output:data());
   return self.output
end

function LRN:updateGradInput(input, gradOutput)
   if not self.gradInput then return end
   self.gradInput:resizeAs(input)

   assert(gradOutput:dim() == 3 or gradOutput:dim() == 4);
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   if not self.LRNDesc then self:resetDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnLRNCrossChannelBackward',
            cudnn.getHandle(), self.LRNDesc[0],
            'CUDNN_LRN_CROSS_CHANNEL_DIM1',
            cudnn.scalar(input, 1),
            self.iDesc[0], self.output:data(),
            self.iDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end

function LRN:clearDesc()
   self.LRNDesc = nil
   self.iDesc = nil
end

function LRN:write(f)
   self:clearDesc()
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function LRN:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_gradOutput')
   return nn.Module.clearState(self)
end
