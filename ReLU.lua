local ReLU, parent = torch.class('cudnn.ReLU','nn.Module')
local ffi = require 'ffi'
local C = cudnn.C
local errcheck = cudnn.errcheck

function ReLU:__init()
   parent.__init(self)
   self.iSize = torch.LongStorage(4):fill(0)   
end

function ReLU:createIODescriptors(input)
   if input:size(1) ~= self.iSize:size(1) or input:size(2) ~= self.iSize:size(2) 
   or input:size(3) ~= self.iSize:size(3)  or input:size(4) ~= self.iSize:size(4) then
      self.gradInput:resizeAs(input)
      self.output:resizeAs(input)
      self.iDesc = cudnn.toDescriptor(input)
      self.oDesc = cudnn.toDescriptor(self.output)
   end
end

function ReLU:updateOutput(input)
   assert(input:dim() == 4 and input:isContiguous());
   self:createIODescriptors(input)
   errcheck('cudnnActivationForward', cudnn.handle[0], 'CUDNN_ACTIVATION_RELU',
            self.iDesc[0], input:data(), 
            self.oDesc[0], self.output:data());
   return self.output
end

function ReLU:updateGradInput(input, gradOutput)
   assert(input:dim() == 4 and input:isContiguous());
   assert(gradOutput:dim() == 4 and gradOutput:isContiguous());
   errcheck('cudnnActivationBackward', cudnn.handle[0], 'CUDNN_ACTIVATION_RELU',
            self.oDesc[0], self.output:data(),
            self.oDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(), 
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end
