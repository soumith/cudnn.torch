local SpatialBatchNormalization, parent = torch.class('cudnn.SpatialBatchNormalization', 'nn.SpatialBatchNormalization')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function SpatialBatchNormalization:__init(nFeature, eps, momentum, affine)
   parent.__init(self, nFeature, eps, momentum, affine)
   self.mode = 'CUDNN_BATCHNORM_SPATIAL'
   self.nFeature = nFeature
   self.save_mean = torch.Tensor(nFeature)
   self.save_std = torch.Tensor(nFeature)
end

function SpatialBatchNormalization:createIODescriptors(input)
   assert(torch.typename(self.weight) == 'torch.CudaTensor' and torch.typename(self.bias) == 'torch.CudaTensor',
          'Only CUDA tensors are supported for cudnn.SpatialBatchNormalization!')
   self.iDesc = cudnn.toDescriptor(input)
   self.sDesc = cudnn.toDescriptor(self.bias:view(1, self.nFeature, 1, 1))
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

function SpatialBatchNormalization:updateOutput(input)
   self:createIODescriptors(input)

   self.output:resizeAs(input)
   self.gradInput:resizeAs(input)

   if self.train then
      errcheck('cudnnBatchNormalizationForwardTraining',
            cudnn.getHandle(), self.mode, one:data(), zero:data(),
            self.iDesc[0], input:data(), self.output:data(),
            self.sDesc[0], self.weight:data(), self.bias:data(),
            self.momentum, self.running_mean:data(), self.running_std:data(), self.eps, save_mean:data(), save_std:data());
   else
      errcheck('cudnnBatchNormalizationForwardInference',
            cudnn.getHandle(), self.mode, one:data(), zero:data(),
            self.iDesc[0], input:data(), self.output:data(),
            self.sDesc[0], self.weight:data(), self.bias:data(),
            self.running_mean:data(), self.running_std:data(), self.eps);
   end
   return self.output
end

function SpatialBatchNormalization:updateGradInput(input, gradOutput)
   assert(gradOutput:isContiguous());
   self:createIODescriptors(input)
   errcheck('cudnnBatchNormalizationBackward',
      cudnn.getHandle(), self.mode, one:data(), zero:data(),
      self.iDesc[0], input:data(), gradOutput:data(), self.gradInput:data(),
                     -- input is bottom, gradOutput is topDiff, self.gradInput is resultBottomDiff
      self.sDesc[0], self.weight:data(), self.gradWeight:data(), self.gradBias:data(),
      self.eps, save_mean:data(), save_std:data());
   return self.gradInput
end

function SpatialBatchNormalization:accGradParameters(input, gradOutput, scale)
end

function SpatialBatchNormalization:write(f)
   self.iDesc = nil
   self.sDesc = nil
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end
