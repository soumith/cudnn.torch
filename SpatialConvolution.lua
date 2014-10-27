local SpatialConvolution, parent = torch.class('cudnn.SpatialConvolution', 'nn.SpatialConvolution')
local ffi = require 'ffi'
local C = cudnn.C
local errcheck = cudnn.errcheck

function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
   self.padW = padW or 0
   self.padH = padH or 0
   self:reset()
   self.iSize = torch.LongStorage(4):fill(0)
end

-- if you change the configuration of the module manually, call this
function SpatialConvolution:resetWeightDescriptors()
   assert(torch.typename(self.weight) == 'torch.CudaTensor', 'Only Cuda supported duh!')
   assert(torch.typename(self.bias) == 'torch.CudaTensor', 'Only Cuda supported duh!')
   -- create filterDescriptor for weight
   self.weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
   errcheck('cudnnCreateFilterDescriptor', self.weightDesc)
   errcheck('cudnnSetFilterDescriptor', self.weightDesc[0], 'CUDNN_DATA_FLOAT',
            self.nOutputPlane, self.nInputPlane, self.kH, self.kW);
   local function destroyWDesc(d)
      errcheck('cudnnDestroyFilterDescriptor', d[0]);
   end
   ffi.gc(self.weightDesc, destroyWDesc)

   -- create descriptor for bias
   self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane, 1, 1))
end

function SpatialConvolution:createIODescriptors(input)
   if not self.iDesc or not self.oDesc or
      input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
   or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
         self.iSize = input:size()
         -- resize gradInput
         if self.gradInput then self.gradInput:resizeAs(input); end
         -- create input descriptor
         self.iDesc = cudnn.toDescriptor(input)
         -- create conv descriptor
         self.convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
         errcheck('cudnnCreateConvolutionDescriptor', self.convDesc)
         errcheck('cudnnSetConvolutionDescriptor', self.convDesc[0], self.iDesc[0],
                  self.weightDesc[0], self.padH, self.padW,
                  self.dH, self.dW, 1, 1, 'CUDNN_CROSS_CORRELATION');
         local function destroyConvDesc(d)
            errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
         end
         ffi.gc(self.convDesc, destroyConvDesc)

         -- create output descriptor and resize output
         local oSize = torch.IntTensor(4):fill(0)
         local oSizeD = oSize:data()
         errcheck('cudnnGetOutputTensor4dDim', self.convDesc[0], 'CUDNN_CONVOLUTION_FWD',
                  oSizeD, oSizeD+1, oSizeD+2, oSizeD+3)
         self.output:resize(oSize:long():storage())
         -- create descriptor for output
         self.oDesc = cudnn.toDescriptor(self.output)
   end
end

function SpatialConvolution:updateOutput(input)
   assert(input:dim() == 4 and input:isContiguous());
   if not self.weightDesc then self:resetWeightDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnConvolutionForward', cudnn.handle[cutorch.getDevice()-1],
            self.iDesc[0], input:data(),
            self.weightDesc[0], self.weight:data(),
            self.convDesc[0], self.oDesc[0], self.output:data(),
            'CUDNN_RESULT_NO_ACCUMULATE');
   local alpha = torch.FloatTensor({1});
   errcheck('cudnnAddTensor4d', cudnn.handle[cutorch.getDevice()-1], 'CUDNN_ADD_SAME_C',
            alpha:data(), self.biasDesc[0], self.bias:data(),
            self.oDesc[0], self.output:data());
   return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
   if not self.gradInput then return end
   assert(input:dim() == 4 and input:isContiguous());
   assert(gradOutput:dim() == 4 and gradOutput:isContiguous());
   if not self.weightDesc then self:resetWeightDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnConvolutionBackwardData', cudnn.handle[cutorch.getDevice()-1],
            self.weightDesc[0], self.weight:data(),
            self.oDesc[0], gradOutput:data(),
            self.convDesc[0],
            self.iDesc[0], self.gradInput:data(),
            'CUDNN_RESULT_NO_ACCUMULATE');
   return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   assert(scale == nil or scale == 1)
   assert(input:dim() == 4 and input:isContiguous());
   assert(gradOutput:dim() == 4 and gradOutput:isContiguous());
   self:createIODescriptors(input)
   if not self.weightDesc then self:resetWeightDescriptors() end
   -- gradBias
   errcheck('cudnnConvolutionBackwardBias', cudnn.handle[cutorch.getDevice()-1],
            self.oDesc[0], gradOutput:data(),
            self.biasDesc[0], self.gradBias:data(),
            'CUDNN_RESULT_ACCUMULATE');
   -- gradWeight
   errcheck('cudnnConvolutionBackwardFilter', cudnn.handle[cutorch.getDevice()-1],
            self.iDesc[0], input:data(),
            self.oDesc[0], gradOutput:data(),
            self.convDesc[0],
            self.weightDesc[0], self.gradWeight:data(),
            'CUDNN_RESULT_ACCUMULATE');

end
--[[
function SpatialConvolution:zeroGradParameters()
   -- gradWeight, gradBias to zero
   local alpha = torch.FloatTensor({0});
   errcheck('cudnnSetTensor4d', self.weightDesc, self.gradWeight:data(), alpha:data());
   errcheck('cudnnSetTensor4d', self.biasDesc, self.gradBias:data(), alpha:data());
end
]]--
