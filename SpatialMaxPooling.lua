local SpatialMaxPooling, parent = torch.class('cudnn.SpatialMaxPooling', 'nn.Module')
local ffi = require 'ffi'
local C = cudnn.C
local errcheck = cudnn.errcheck

function SpatialMaxPooling:__init(kW, kH, dW, dH)
   parent.__init(self)
   self.kW = kW
   self.kH = kH
   self.dW = dW or kW
   self.dH = dH or kW
   self.iSize = torch.LongStorage(4):fill(0)
end

function SpatialMaxPooling:resetPoolDescriptors()
   -- create pooling descriptor
   self.poolDesc = ffi.new('struct cudnnPoolingStruct*[1]')
   errcheck('cudnnCreatePoolingDescriptor', self.poolDesc)
   errcheck('cudnnSetPoolingDescriptor', self.poolDesc[0], 'CUDNN_POOLING_MAX',
            self.kH, self.kW, self.dH, self.dW);
   local function destroyPoolDesc(d) 
      errcheck('cudnnDestroyPoolingDescriptor', d[0]);
   end
   ffi.gc(self.poolDesc, destroyPoolDesc)
end

function SpatialMaxPooling:createIODescriptors(input)
   if not self.iDesc or not self.oDesc or 
      input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
   or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
      self.iSize = input:size()
      -- resize gradInput
      self.gradInput:resizeAs(input)
      -- resize output
      local oW = math.floor((input:size(4) - self.kW)/self.dW + 1)
      local oH = math.floor((input:size(3) - self.kH)/self.dH + 1)
      self.output:resize(input:size(1), input:size(2), oH, oW)

      -- create input/output descriptor
      self.iDesc = cudnn.toDescriptor(input)
      self.oDesc = cudnn.toDescriptor(self.output)
   end
end

function SpatialMaxPooling:updateOutput(input)
   assert(input:dim() == 4 and input:isContiguous());
   if not self.poolDesc then self:resetPoolDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnPoolingForward', cudnn.handle[cutorch.getDevice()-1], self.poolDesc[0],
            self.iDesc[0], input:data(), 
            self.oDesc[0], self.output:data());
   return self.output
end

function SpatialMaxPooling:updateGradInput(input, gradOutput)
   assert(input:dim() == 4 and input:isContiguous());
   assert(gradOutput:dim() == 4 and gradOutput:isContiguous());
   if not self.poolDesc then self:resetPoolDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnPoolingBackward', cudnn.handle[cutorch.getDevice()-1], self.poolDesc[0],
            self.oDesc[0], self.output:data(),
            self.oDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(), 
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end

