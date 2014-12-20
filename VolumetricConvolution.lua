local VolumetricConvolution, parent
   = torch.class('cudnn.VolumetricConvolution', 'nn.VolumetricConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function VolumetricConvolution:__init(nInputPlane, nOutputPlane,
                               kT, kW, kH,
                               dT, dW, dH,
                               padT, padW, padH)
   parent.__init(self, nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH)
   self.padT = padT or 0
   self.padW = padW or 0
   self.padH = padH or 0
   self:reset()
   self.iSize = torch.LongStorage(5):fill(0)
end

-- if you change the configuration of the module manually, call this
function VolumetricConvolution:resetWeightDescriptors()
   assert(torch.typename(self.weight) == 'torch.CudaTensor',
          'Only Cuda supported duh!')
   assert(torch.typename(self.bias) == 'torch.CudaTensor',
          'Only Cuda supported duh!')
   -- create filterDescriptor for weight
   self.weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
   errcheck('cudnnCreateFilterDescriptor', self.weightDesc)
   local desc = torch.IntTensor({self.nOutputPlane, self.nInputPlane,
                             self.kT, self.kH, self.kW})
   errcheck('cudnnSetFilterNdDescriptor', self.weightDesc[0],
            'CUDNN_DATA_FLOAT', 5,
            desc:data());
   local function destroyWDesc(d)
      errcheck('cudnnDestroyFilterDescriptor', d[0]);
   end
   ffi.gc(self.weightDesc, destroyWDesc)

   -- create descriptor for bias
   self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,
                                                     1, 1))
end

function VolumetricConvolution:createIODescriptors(input)
   local batch = true
   if input:dim() == 4 then
      input = input:view(1, input:size(1), input:size(2),
                         input:size(3), input:size(4))
      batch = false
   end
   assert(input:dim() == 5 and input:isContiguous());
   if not self.iDesc or not self.oDesc or
      input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
   or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4]
   or input:size(5) ~= self.iSize[5] then
         self.iSize = input:size()
         -- resize gradInput
         if self.gradInput then self.gradInput:resizeAs(input); end
         -- create input descriptor
         self.iDesc = cudnn.toDescriptor(input)
         -- create conv descriptor
         self.convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
         errcheck('cudnnCreateConvolutionDescriptor', self.convDesc)
         local pad = torch.IntTensor({self.padT, self.padH, self.padW})
         local stride = torch.IntTensor({self.dT, self.dH, self.dW})
         local upscale = torch.IntTensor({1,1,1})
         errcheck('cudnnSetConvolutionNdDescriptor', self.convDesc[0],
                  3, pad:data(),
                  stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION');
         local function destroyConvDesc(d)
            errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
         end
         ffi.gc(self.convDesc, destroyConvDesc)

         -- create output descriptor and resize output
         local oSize = torch.IntTensor(5)
         local oSizeD = oSize:data()
         errcheck('cudnnGetConvolutionNdForwardOutputDim',
                  self.convDesc[0], self.iDesc[0],
                  self.weightDesc[0], 5, oSizeD)
         self.output:resize(oSize:long():storage())
         -- create descriptor for output
         self.oDesc = cudnn.toDescriptor(self.output)
         self.oDescBias = cudnn.toDescriptor(
            self.output:view(self.output:size(1),
                             self.output:size(2),
                             self.output:size(3)*self.output:size(4),
                             self.output:size(5)))

         -- create forwardAlgorithm descriptors for
         local algType = ffi.new("cudnnConvolutionFwdAlgo_t[?]", 1)
         errcheck('cudnnGetConvolutionForwardAlgorithm',
                  cudnn.handle[cutorch.getDevice()-1],
                  self.iDesc[0], self.weightDesc[0], self.convDesc[0],
                  self.oDesc[0], 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST',
                     -1, algType)
         self.algType = algType
         local bufSize = torch.LongTensor(1)
         errcheck('cudnnGetConvolutionForwardWorkspaceSize',
                  cudnn.handle[cutorch.getDevice()-1],
                  self.iDesc[0], self.weightDesc[0],
                  self.convDesc[0], self.oDesc[0],
                  algType[0], bufSize:data())
         self.extraBuffer = self.extraBuffer or input.new(1)
         if bufSize[1] ~= 0 then self.extraBuffer:resize(bufSize[1]) end

         if not batch then
            self.gradInput = self.gradInput:view(self.gradInput:size(2),
                                                 self.gradInput:size(3),
                                                 self.gradInput:size(4),
                                                 self.gradInput:size(5))
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4),
                                           self.output:size(5))
         end
   end
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

function VolumetricConvolution:updateOutput(input)
   if not self.weightDesc then self:resetWeightDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnConvolutionForward', cudnn.handle[cutorch.getDevice()-1],
            one:data(),
            self.iDesc[0], input:data(),
            self.weightDesc[0], self.weight:data(),
            self.convDesc[0], self.algType[0],
            self.extraBuffer:data(), self.extraBuffer:nElement(),
            zero:data(),
            self.oDesc[0], self.output:data());
   errcheck('cudnnAddTensor', cudnn.handle[cutorch.getDevice()-1],
            'CUDNN_ADD_SAME_C', one:data(),
            self.biasDesc[0], self.bias:data(), one:data(),
            self.oDescBias[0], self.output:data());
   return self.output
end

function VolumetricConvolution:updateGradInput(input, gradOutput)
   if not self.gradInput then return end
   assert((gradOutput:dim() == 4 or gradOutput:dim() == 5)
         and gradOutput:isContiguous());
   if not self.weightDesc then self:resetWeightDescriptors() end
   self:createIODescriptors(input)
   errcheck('cudnnConvolutionBackwardData', cudnn.handle[cutorch.getDevice()-1],
            one:data(),
            self.weightDesc[0], self.weight:data(),
            self.oDesc[0], gradOutput:data(),
            self.convDesc[0],
            zero:data(),
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end

function VolumetricConvolution:accGradParameters(input, gradOutput, scale)
   self.scaleT = self.scaleT or torch.FloatTensor(1):fill(1.0)
   -- this line forces this member to always be on CPU (needed for cudnn)
   self.scaleT = self.scaleT:float()

   scale = scale or 1.0
   self.scaleT[1] = scale
   assert((gradOutput:dim() == 4 or gradOutput:dim() == 5)
         and gradOutput:isContiguous());
   self:createIODescriptors(input)
   if not self.weightDesc then self:resetWeightDescriptors() end
   -- gradBias
   errcheck('cudnnConvolutionBackwardBias', cudnn.handle[cutorch.getDevice()-1],
            self.scaleT:data(),
            self.oDescBias[0], gradOutput:data(),
            one:data(),
            self.biasDesc[0], self.gradBias:data());
   -- gradWeight
   errcheck('cudnnConvolutionBackwardFilter',
            cudnn.handle[cutorch.getDevice()-1],
            self.scaleT:data(),
            self.iDesc[0], input:data(),
            self.oDesc[0], gradOutput:data(),
            self.convDesc[0],
            one:data(),
            self.weightDesc[0], self.gradWeight:data());

end
