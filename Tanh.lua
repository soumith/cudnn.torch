local Tanh, parent = torch.class('cudnn.Tanh','nn.Module')
local ffi = require 'ffi'
local C = cudnn.C
local errcheck = cudnn.errcheck

function Tanh:__init()
   parent.__init(self)
   self.iSize = torch.LongStorage(4):fill(0)   
end

function Tanh:createIODescriptors(input)
   if not self.iDesc or not self.oDesc or 
      input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
   or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
      self.iSize = input:size()
      self.gradInput:resizeAs(input)
      self.output:resizeAs(input)
      self.iDesc = cudnn.toDescriptor(input)
      self.oDesc = cudnn.toDescriptor(self.output)
   end
end

function Tanh:updateOutput(input)
   assert(input:dim() == 4 and input:isContiguous());
   self:createIODescriptors(input)
   errcheck('cudnnActivationForward', cudnn.handle[cutorch.getDevice()-1], 'CUDNN_ACTIVATION_TANH',
            self.iDesc[0], input:data(), 
            self.oDesc[0], self.output:data());
   return self.output
end

function Tanh:updateGradInput(input, gradOutput)
   assert(input:dim() == 4 and input:isContiguous());
   assert(gradOutput:dim() == 4 and gradOutput:isContiguous());
   self:createIODescriptors(input)
   errcheck('cudnnActivationBackward', cudnn.handle[cutorch.getDevice()-1], 'CUDNN_ACTIVATION_TANH',
            self.oDesc[0], self.output:data(),
            self.oDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(), 
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end
