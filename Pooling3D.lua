local Pooling, parent = torch.class('cudnn._Pooling3D', 'nn.Module')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function Pooling:__init(kT, kW, kH, dT, dW, dH, padT, padW, padH)
   parent.__init(self)
   self.kT = kT
   self.kW = kW
   self.kH = kH
   self.dT = dT or kT
   self.dW = dW or kW
   self.dH = dH or kH
   self.padT = padT or 0
   self.padW = padW or 0
   self.padH = padH or 0
   self.iSize = torch.LongStorage(5):fill(0)
   self.ceil_mode = false
end

function Pooling:ceil()
   self.ceil_mode = true
   return self
end

function Pooling:floor()
   self.ceil_mode = false
   return self
end

function Pooling:resetPoolDescriptors()
   -- create pooling descriptor
   self.padT = self.padT or 0
   self.padW = self.padW or 0
   self.padH = self.padH or 0
   self.poolDesc = ffi.new('struct cudnnPoolingStruct*[1]')
   errcheck('cudnnCreatePoolingDescriptor', self.poolDesc)
   local ker = torch.IntTensor({self.kT, self.kH, self.kW})
   local str = torch.IntTensor({self.dT, self.dH, self.dW})
   local pad = torch.IntTensor({self.padT, self.padH, self.padW})
   errcheck('cudnnSetPoolingNdDescriptor', self.poolDesc[0], self.mode, 'CUDNN_PROPAGATE_NAN', 3,
            ker:data(), pad:data(), str:data());
   local function destroyPoolDesc(d)
      errcheck('cudnnDestroyPoolingDescriptor', d[0]);
   end
   ffi.gc(self.poolDesc, destroyPoolDesc)
end

function Pooling:createIODescriptors(input)
   assert(self.mode, 'mode is not set. (trying to use base class?)');
   local batch = true
   if input:dim() == 4 then
       input = input:view(1, input:size(1), input:size(2), input:size(3), input:size(4))
      batch = false
   end
   assert(input:dim() == 5 and input:isContiguous());
   if not self.iDesc or not self.oDesc or
      input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
   or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4]
   or input:size(5) ~= self.iSize[5] then
      self.iSize = input:size()
      -- resize output
      local oW, oH, oT
      if self.ceil_mode then
         oW = math.ceil((input:size(5)+self.padW*2 - self.kW)/self.dW + 1)
         oH = math.ceil((input:size(4)+self.padH*2 - self.kH)/self.dH + 1)
         oT = math.ceil((input:size(3)+self.padT*2 - self.kT)/self.dT + 1)
      else
         oW = math.floor((input:size(5)+self.padW*2 - self.kW)/self.dW + 1)
         oH = math.floor((input:size(4)+self.padH*2 - self.kH)/self.dH + 1)
         oT = math.floor((input:size(3)+self.padT*2 - self.kT)/self.dT + 1)
      end
      self.output:resize(input:size(1), input:size(2), oT, oH, oW)

      -- create input/output descriptor
      self.iDesc = cudnn.toDescriptor(input)
      self.oDesc = cudnn.toDescriptor(self.output)
      if not batch then
         self.output = self.output:view(self.output:size(2),
                                        self.output:size(3),
                                        self.output:size(4),
                                        self.output:size(5))
      end
   end
end

function Pooling:updateOutput(input)
   if not self.poolDesc then self:resetPoolDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnPoolingForward', cudnn.getHandle(),
            self.poolDesc[0],
            cudnn.scalar(input, 1),
            self.iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            self.oDesc[0], self.output:data());
   return self.output
end

function Pooling:updateGradInput(input, gradOutput)
   if not self.gradInput then return end
   self.gradInput:resizeAs(input)

   assert(gradOutput:dim() == 4 or gradOutput:dim() == 5);
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   if not self.poolDesc then self:resetPoolDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnPoolingBackward',
            cudnn.getHandle(), self.poolDesc[0],
            cudnn.scalar(input, 1),
            self.oDesc[0], self.output:data(),
            self.oDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end

function Pooling:clearDesc()
   self.poolDesc = nil
   self.iDesc = nil
   self.oDesc = nil
end

function Pooling:write(f)
   self:clearDesc()
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function Pooling:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_gradOutput')
   return parent.clearState(self)
end
