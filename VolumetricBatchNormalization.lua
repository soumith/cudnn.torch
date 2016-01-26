local VolumetricBatchNormalization, parent = torch.class('cudnn.VolumetricBatchNormalization', 'nn.Module')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function VolumetricBatchNormalization:__init(nFeature, eps, momentum, affine)
   parent.__init(self)
   assert(nFeature and type(nFeature) == 'number',
       'Missing argument #1: Number of feature planes. ')
   assert(nFeature ~= 0, 'To set affine=false call BatchNormalization'
     .. '(nFeature,  eps, momentum, false) ')
   assert(affine == nil or affine == true, 'only affine supported')

   self.mode = 'CUDNN_BATCHNORM_SPATIAL'
   self.nFeature = nFeature
   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1
   self.save_mean = torch.Tensor(nFeature)
   self.save_std = torch.Tensor(nFeature)
   self.running_mean = torch.zeros(nFeature)
   self.running_std = torch.ones(nFeature)
   self.weight = torch.Tensor(nFeature)
   self.bias = torch.Tensor(nFeature)
   self.gradWeight = torch.Tensor(nFeature)
   self.gradBias = torch.Tensor(nFeature)
   self:reset()
end

function VolumetricBatchNormalization:createIODescriptors(input)
   assert(input:dim() == 5)
   assert(torch.typename(self.weight) == 'torch.CudaTensor' and torch.typename(self.bias) == 'torch.CudaTensor',
          'Only CUDA tensors are supported for cudnn.VolumetricBatchNormalization!')
   if not self.iDesc or not self.oDesc or
      input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
   or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] 
   or input:size(5) ~= self.iSize[5] then
      self.iSize = input:size()
      self.output:resizeAs(input)
      self.gradInput:resizeAs(input)
      self.iDesc = cudnn.toDescriptor(input)
      self.oDesc = cudnn.toDescriptor(self.output)
      self.sDesc = cudnn.toDescriptor(self.bias:view(1, self.nFeature, 1, 1, 1)) 
   end
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

function VolumetricBatchNormalization:reset()
   self.weight:uniform()
   self.bias:zero()
end

function VolumetricBatchNormalization:updateOutput(input)
   self:createIODescriptors(input)

   if self.train then
      errcheck('cudnnBatchNormalizationForwardTraining',
            cudnn.getHandle(), self.mode, one:data(), zero:data(),
            self.iDesc[0], input:data(), self.oDesc[0], self.output:data(),
            self.sDesc[0], self.weight:data(), self.bias:data(),
            self.momentum, self.running_mean:data(), self.running_std:data(), self.eps, self.save_mean:data(), self.save_std:data());
   else
      errcheck('cudnnBatchNormalizationForwardInference',
            cudnn.getHandle(), self.mode, one:data(), zero:data(),
            self.iDesc[0], input:data(), self.oDesc[0], self.output:data(),
            self.sDesc[0], self.weight:data(), self.bias:data(),
            self.running_mean:data(), self.running_std:data(), self.eps);
   end
   return self.output
end

function VolumetricBatchNormalization:updateGradInput(input, gradOutput)
   assert(gradOutput:isContiguous());
   self:createIODescriptors(input)
   errcheck('cudnnBatchNormalizationBackward',
      cudnn.getHandle(), self.mode, one:data(), zero:data(),
      self.iDesc[0], input:data(), self.iDesc[0], gradOutput:data(), self.iDesc[0], self.gradInput:data(),
                     -- input is bottom, gradOutput is topDiff, self.gradInput is resultBottomDiff
      self.sDesc[0], self.weight:data(), self.gradWeight:data(), self.gradBias:data(),
      self.eps, self.save_mean:data(), self.save_std:data());
   return self.gradInput
end

function VolumetricBatchNormalization:accGradParameters(input, gradOutput, scale)
end

function VolumetricBatchNormalization:write(f)
   self.iDesc = nil
   self.oDesc = nil
   self.sDesc = nil
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end
