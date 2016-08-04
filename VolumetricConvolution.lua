local VolumetricConvolution, parent
   = torch.class('cudnn.VolumetricConvolution', 'nn.VolumetricConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck
local algo = require 'cudnn.algo'

local Convolution = cudnn.SpatialConvolution

function VolumetricConvolution:__init(nInputPlane, nOutputPlane,
                                      kT, kW, kH, dW, dH, padW, padH)
   self.nDim = 5
   self.kT = kT
   Convolution.__init(self,nInputPlane, nOutputPlane,
                      kW, kH, dW, dH, padW, padH, 1)
   return self
end
-- if you change the configuration of the module manually, call this
function VolumetricConvolution:resetWeightDescriptors()
   local desc = torch.IntTensor({self.nOutputPlane, self.nInputPlane,
                             self.kT, self.kH, self.kW})
   Convolution.resetWeightDescriptors(self, desc)
end

function VolumetricConvolution:fastest(mode)
   return Convolution.fastest(self)
end

function VolumetricConvolution:setMode(fmode, bdmode, bwmode)
   return Convolution.setMode(self, fmode, bdmode, bwmode)
end

function VolumetricConvolution:resetMode()
   return Convolution.resetMode(self)
end

function VolumetricConvolution:createIODescriptors(input)
   if input:dim() == 4 then
      input = input:view(1, input:size(1), input:size(2),
                         input:size(3), input:size(4))
      batch = false
   end
   if Convolution.checkInputChanged(self, input) then
         -- create input descriptor
         self.iDesc = cudnn.toDescriptor(input)
         -- create conv descriptor
         self.convDesc = cudnn.createDescriptors(1, 'struct cudnnConvolutionStruct*[?]',
                                                 'cudnnCreateConvolutionDescriptor', 'cudnnDestroyConvolutionDescriptor')
         local pad = torch.IntTensor({self.padT, self.padH, self.padW})
         local stride = torch.IntTensor({self.dT, self.dH, self.dW})
         local upscale = torch.IntTensor({1,1,1})
         errcheck('cudnnSetConvolutionNdDescriptor', self.convDesc[0],
                  3, pad:data(),
                  stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
                  cudnn.configmap(torch.type(self.weight)));
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

         algo.prepareHash(self, input, output)

         if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4),
                                           self.output:size(5))
         end
   end
end

function VolumetricConvolution:updateOutput(input)
   return Convolution:updateOutput(input)
end

function VolumetricConvolution:updateGradInput(input, gradOutput)
   return Convolution:updateGradInput(input)
end

function VolumetricConvolution:accGradParameters(input, gradOutput, scale)
   return Convolution:accGradParameters(input, gradOutput, scale)
end

function VolumetricConvolution:clearDesc()
   Convolution:clearDesc()
end

function VolumetricConvolution:write(f)
   Convolution:write(f)
end

function VolumetricConvolution:clearState()
   return Convolution:clearState()
end

return VolumetricConvolution
