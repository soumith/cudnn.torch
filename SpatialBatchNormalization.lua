local SpatialBatchNormalization, parent = torch.class('cudnn.SpatialBatchNormalization', 'nn.SpatialBatchNormalization')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

SpatialBatchNormalization.__version = 2

function SpatialBatchNormalization:createIODescriptors(input)
   assert(input:dim() == 4)
   assert(torch.typename(self.weight) == 'torch.CudaTensor' and torch.typename(self.bias) == 'torch.CudaTensor',
          'Only CUDA tensors are supported for cudnn.SpatialBatchNormalization!')
   local nFeature = self.running_mean:numel()
   self.mode = 'CUDNN_BATCHNORM_SPATIAL'
   self.save_mean = self.save_mean or input.new()
   self.save_std  = self.save_std  or input.new()
   if not self.iDesc or not self.oDesc or
      input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
   or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
      self.iSize = input:size()
      self.output:resizeAs(input)
      self.gradInput:resizeAs(input)
      self.iDesc = cudnn.toDescriptor(input)
      self.oDesc = cudnn.toDescriptor(self.output)
      self.sDesc = cudnn.toDescriptor(self.bias:view(1, nFeature, 1, 1))
   end
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});
local scaleTens = torch.FloatTensor(1);

function SpatialBatchNormalization:updateOutput(input)
   self:createIODescriptors(input)

   if self.train then
      errcheck('cudnnBatchNormalizationForwardTraining',
            cudnn.getHandle(), self.mode, one:data(), zero:data(),
            self.iDesc[0], input:data(), self.oDesc[0], self.output:data(),
            self.sDesc[0], self.weight:data(), self.bias:data(),
            self.momentum, self.running_mean:data(), self.running_var:data(), self.eps, self.save_mean:data(), self.save_std:data());
   else
      errcheck('cudnnBatchNormalizationForwardInference',
            cudnn.getHandle(), self.mode, one:data(), zero:data(),
            self.iDesc[0], input:data(), self.oDesc[0], self.output:data(),
            self.sDesc[0], self.weight:data(), self.bias:data(),
            self.running_mean:data(), self.running_var:data(), self.eps);
   end
   return self.output
end

local function backward(self,input,gradOutput, scale)
   assert(gradOutput:isContiguous())
   self:createIODescriptors(input)
   scale = scale or 1
   scaleTens:fill(scale)
   errcheck('cudnnBatchNormalizationBackward',
      cudnn.getHandle(), self.mode, one:data(), zero:data(), scaleTens:data(), one:data(),
      self.iDesc[0], input:data(), self.iDesc[0], gradOutput:data(), self.iDesc[0], self.gradInput:data(),
                     -- input is bottom, gradOutput is topDiff, self.gradInput is resultBottomDiff
      self.sDesc[0], self.weight:data(), self.gradWeight:data(), self.gradBias:data(),
      self.eps, self.save_mean:data(), self.save_std:data());
   return self.gradInput
end

function SpatialBatchNormalization:updateGradInput(input, gradOutput, scale)
-- will in fact update gradWeight and gradBias too, accGradParameters call is empty
  return backward(self, input,gradOutput, scale)
end


function SpatialBatchNormalization:backward(input, gradOutput, scale)
  return backward(self, input,gradOutput, scale)
end

function SpatialBatchNormalization:accGradParameters(input, gradOutput, scale)
end

function SpatialBatchNormalization:clearDesc()
   self.iDesc = nil
   self.oDesc = nil
   self.sDesc = nil
end

function SpatialBatchNormalization:write(f)
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function SpatialBatchNormalization:clearState()
   self:clearDesc()
   return nn.Module.clearState(self)
end
