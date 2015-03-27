local SpatialConvolution, parent =
   torch.class('cudnn.SpatialConvolution', 'nn.SpatialConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function SpatialConvolution:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH, groups)
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
   self.padW = padW or 0
   self.padH = padH or 0
   self.groups = groups or 1
   assert(nInputPlane % self.groups == 0,
          'nInputPlane should be divisible by nGroups')
   assert(nOutputPlane % self.groups == 0,
          'nOutputPlane should be divisible by nGroups')
   self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kW, kH)
   self:reset()
   self.iSize = torch.LongStorage(4):fill(0)
end

-- if you change the configuration of the module manually, call this
function SpatialConvolution:resetWeightDescriptors()
   assert(torch.typename(self.weight) == 'torch.CudaTensor',
          'Only Cuda supported duh!')
   assert(torch.typename(self.bias) == 'torch.CudaTensor',
          'Only Cuda supported duh!')
   -- for compatibility
   self.groups = self.groups or 1
   -- create filterDescriptor for weight
   self.weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
   errcheck('cudnnCreateFilterDescriptor', self.weightDesc)
   local desc = torch.IntTensor({self.nOutputPlane/self.groups,
                             self.nInputPlane/self.groups,
                             self.kH, self.kW})
   errcheck('cudnnSetFilterNdDescriptor', self.weightDesc[0],
            'CUDNN_DATA_FLOAT', 4,
            desc:data());
   local function destroyWDesc(d)
      errcheck('cudnnDestroyFilterDescriptor', d[0]);
   end
   ffi.gc(self.weightDesc, destroyWDesc)

   -- create descriptor for bias
   local bias_slice = {{}, {1,self.nOutputPlane/self.groups}, {}, {}}
   self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1)[bias_slice])
end

function SpatialConvolution:fastest(mode)
   mode = mode or true
   self.fastest_mode = mode
   return self
end

function SpatialConvolution:createIODescriptors(input)
   local batch = true
   if input:dim() == 3 then
      input = input:view(1, input:size(1), input:size(2), input:size(3))
      batch = false
   end
   assert(input:dim() == 4 and input:isContiguous());
   if not self.iDesc or not self.oDesc or
      input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
   or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
         self.iSize = input:size()
         -- resize gradInput
         if self.gradInput then self.gradInput:resizeAs(input); end
         assert(self.nInputPlane == input:size(2), 'input has to contain: '
                   .. self.nInputPlane
                   .. ' feature maps, but received input of size: '
                   .. input:size(1) .. ' x ' .. input:size(2) ..
                   ' x ' .. input:size(3) .. ' x ' .. input:size(4))
         -- create input descriptor
         local input_slice = {{},{1,self.nInputPlane/self.groups},{},{}}
         self.iDesc = cudnn.toDescriptor(input[input_slice])
         -- create conv descriptor
         self.convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
         errcheck('cudnnCreateConvolutionDescriptor', self.convDesc)
         local pad = torch.IntTensor({self.padH, self.padW})
         local stride = torch.IntTensor({self.dH, self.dW})
         local upscale = torch.IntTensor({1,1})
         errcheck('cudnnSetConvolutionNdDescriptor', self.convDesc[0],
                  2, pad:data(),
                  stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION');
         local function destroyConvDesc(d)
            errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
         end
         ffi.gc(self.convDesc, destroyConvDesc)

         -- create output descriptor and resize output
         local oSize = torch.IntTensor(4)
         local oSizeD = oSize:data()
         errcheck('cudnnGetConvolutionNdForwardOutputDim',
                  self.convDesc[0], self.iDesc[0],
                  self.weightDesc[0], 4, oSizeD)
         oSize[2] = oSize[2] * self.groups
         self.output:resize(oSize:long():storage())
         -- create descriptor for output
         local output_slice = {{},{1,self.nOutputPlane/self.groups},{},{}}
         self.oDesc = cudnn.toDescriptor(self.output[output_slice])

         -- create forwardAlgorithm descriptors for
         local algType = ffi.new("cudnnConvolutionFwdAlgo_t[?]", 1)
	 local algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
	 local algWorkspaceLimit = self.nInputPlane * self.kH * self.kW * 4 -- 4 = sizeof int.
	 if self.fastest_mode then algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST' end
         errcheck('cudnnGetConvolutionForwardAlgorithm',
                  cudnn.handle[cutorch.getDevice()-1],
                  self.iDesc[0], self.weightDesc[0],
                  self.convDesc[0], self.oDesc[0],
                  algSearchMode, algWorkspaceLimit, algType)
         self.algType = algType
         local bufSize = torch.LongTensor(1)
         errcheck('cudnnGetConvolutionForwardWorkspaceSize',
                  cudnn.handle[cutorch.getDevice()-1],
                  self.iDesc[0], self.weightDesc[0],
                  self.convDesc[0], self.oDesc[0],
                  algType[0], bufSize:data())
         self.extraBuffer = self.extraBuffer or input.new(1)
         if bufSize[1] ~= 0 then self.extraBuffer:resize(bufSize[1]) end

         -- create offsets for groups
         self.input_offset = self.nInputPlane/self.groups*input:size(3)*input:size(4)
         self.output_offset = self.nOutputPlane/self.groups*oSize[3]*oSize[4]
         self.weight_offset =
            self.nInputPlane/self.groups*self.nOutputPlane/self.groups*self.kW*self.kH
         self.bias_offset = self.nOutputPlane/self.groups

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

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

function SpatialConvolution:updateOutput(input)
   if not self.weightDesc then self:resetWeightDescriptors() end
   self:createIODescriptors(input)
   for g=0,self.groups-1 do
      errcheck('cudnnConvolutionForward', cudnn.handle[cutorch.getDevice()-1],
               one:data(),
               self.iDesc[0], input:data() + g*self.input_offset,
               self.weightDesc[0], self.weight:data() + g*self.weight_offset,
               self.convDesc[0], self.algType[0],
               self.extraBuffer:data(), self.extraBuffer:nElement(),
               zero:data(),
               self.oDesc[0], self.output:data() + g*self.output_offset);
      errcheck('cudnnAddTensor', cudnn.handle[cutorch.getDevice()-1],
               'CUDNN_ADD_SAME_C',
               one:data(), self.biasDesc[0], self.bias:data() + g*self.bias_offset,
               one:data(), self.oDesc[0], self.output:data() + g*self.output_offset);
   end
   return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
   if not self.gradInput then return end
   assert((gradOutput:dim() == 3 or gradOutput:dim() == 4)
         and gradOutput:isContiguous());
   if not self.weightDesc then self:resetWeightDescriptors() end
   self:createIODescriptors(input)
   for g=0,self.groups-1 do
      errcheck('cudnnConvolutionBackwardData', cudnn.handle[cutorch.getDevice()-1],
               one:data(),
               self.weightDesc[0], self.weight:data() + g*self.weight_offset,
               self.oDesc[0], gradOutput:data() + g*self.output_offset,
               self.convDesc[0],
               zero:data(),
               self.iDesc[0], self.gradInput:data() + g*self.input_offset);
   end
   return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   self.scaleT = self.scaleT or torch.FloatTensor(1):fill(1.0)
   -- this line forces this member to always be on CPU (needed for cudnn)
   self.scaleT = self.scaleT:float()
   scale = scale or 1.0
   self.scaleT[1] = scale
   assert((gradOutput:dim() == 3 or gradOutput:dim() == 4)
         and gradOutput:isContiguous());
   self:createIODescriptors(input)
   if not self.weightDesc then self:resetWeightDescriptors() end
   for g=0,self.groups-1 do
      -- gradBias
      errcheck('cudnnConvolutionBackwardBias', cudnn.handle[cutorch.getDevice()-1],
               self.scaleT:data(),
               self.oDesc[0], gradOutput:data() + g*self.output_offset,
               one:data(),
               self.biasDesc[0], self.gradBias:data() + g*self.bias_offset);
      -- gradWeight
      errcheck('cudnnConvolutionBackwardFilter', cudnn.handle[cutorch.getDevice()-1],
               self.scaleT:data(),
               self.iDesc[0], input:data() + g*self.input_offset,
               self.oDesc[0], gradOutput:data() + g*self.output_offset,
               self.convDesc[0],
               one:data(),
               self.weightDesc[0], self.gradWeight:data() + g*self.weight_offset);
   end
end

--[[
function SpatialConvolution:zeroGradParameters()
   -- gradWeight, gradBias to zero
   errcheck('cudnnSetTensor', cudnn.handle[cutorch.getDevice()-1],
            self.weightDesc, self.gradWeight:data(), zero:data());
   errcheck('cudnnSetTensor', cudnn.handle[cutorch.getDevice()-1],
            self.biasDesc, self.gradBias:data(), zero:data());
end
]]--
