local DivisiveNorm, parent = torch.class('cudnn.SpatialDivisiveNormalization', 'nn.Module')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function DivisiveNorm:__init(size, alpha, beta, K)
   parent.__init(self)
   self.size = size or 5
   self.alpha = alpha or 1e-4
   self.beta = beta or 0.75
   self.K = K or 2.0
   assert(self.size >= 1 and self.size <= 16, "size has to be between 1 and 16")
   assert(self.K >= 1e-5, "K has to be greater than 1e-5")
   assert(self.beta >= 0.01, "Beta has to be > 0.01")
end

function DivisiveNorm:resetDescriptors()
   -- create DivisiveNorm descriptor
   self.DivisiveNormDesc = ffi.new('struct cudnnDivisiveNormDescriptor_t*[1]')
   errcheck('cudnnCreateDivisiveNormDescriptor', self.DivisiveNormDesc)
   errcheck('cudnnSetDivisiveNormDescriptor', self.DivisiveNormDesc[0], self.size,
            self.alpha, self.beta, self.K);
   local function destroyDesc(d)
      errcheck('cudnnDestroyDivisiveNormDescriptor', d[0]);
   end
   ffi.gc(self.DivisiveNormDesc, destroyDesc)
end

function DivisiveNorm:createIODescriptors(input)
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
      self.gradInput:resizeAs(input)
      self.output:resizeAs(input)

      -- create input/output descriptor
      self.iDesc = cudnn.toDescriptor(input)
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




function DivisiveNorm:updateOutput(input)
   if not self.DivisiveNormDesc then self:resetPoolDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnDivisiveNormCrossChannelForward', cudnn.getHandle(),
            self.DivisiveNormDesc[0],
            'CUDNN_DivisiveNorm_CROSS_CHANNEL_DIM1',
            cudnn.scalar(input, 1),
            self.iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            self.iDesc[0], self.output:data());
   return self.output
end

function DivisiveNorm:updateGradInput(input, gradOutput)
   assert(gradOutput:dim() == 3 or gradOutput:dim() == 4);
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   if not self.DivisiveNormDesc then self:resetPoolDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnDivisiveNormCrossChannelBackward',
            cudnn.getHandle(), self.DivisiveNormDesc[0],
            'CUDNN_DivisiveNorm_CROSS_CHANNEL_DIM1',
            cudnn.scalar(input, 1),
            self.iDesc[0], self.output:data(),
            self.iDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end

function DivisiveNorm:write(f)
   self.DivisiveNormDesc = nil
   self.iDesc = nil
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function DivisiveNorm:clearState()
   self._gradOutput = nil
   return parent.clearState(self)
end
