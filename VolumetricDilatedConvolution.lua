local VolumetricDilatedConvolution, parent
   = torch.class('cudnn.VolumetricDilatedConvolution', 'cudnn.VolumetricConvolution')
local ffi = require 'ffi'
local find = require 'cudnn.find'

local Convolution = cudnn.SpatialConvolution

function VolumetricDilatedConvolution:__init(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH)
   parent.__init(self, nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH)

   self.dilationT = dilationT or 1
   self.dilationW = dilationW or 1
   self.dilationH = dilationH or 1
end


function VolumetricDilatedConvolution:createIODescriptors(input)
   if input:dim() == 4 then
      input = input:view(1, input:size(1), input:size(2),
                         input:size(3), input:size(4))
      batch = false
   end
   if Convolution.checkInputChanged(self, input) then
         -- create input descriptor
         self.iDesc = cudnn.toDescriptor(input)
         -- create conv descriptor
         self.pad = {self.padT, self.padH, self.padW}
         self.stride = {self.dT, self.dH, self.dW}
         self.dilation = {self.dilationT, self.dilationH, self.dilationW}

         local mathtype=cudnn.configmap(torch.type(self.weight))
         -- 3D convolutions do not work in 16 bits
         if mathtype == 'CUDNN_DATA_HALF' then
            mathtype = 'CUDNN_DATA_FLOAT'
         end
         self.convDescData = {
            padA = self.pad,
            filterStrideA = self.stride,
            dilationA = self.dilation,
            dataType = mathtype
         }
         self.convDesc = cudnn.setConvolutionDescriptor(self.convDescData)

         local oSize = torch.IntTensor(5)
         cudnn.errcheck('cudnnGetConvolutionNdForwardOutputDim',
                  self.convDesc[0], self.iDesc[0],
                  self.weightDesc[0], 5, oSize:data())
         self.output:resize(oSize:long():storage())
         -- create descriptor for output
         self.oDesc = cudnn.toDescriptor(self.output)
         self.oDescForBias = cudnn.toDescriptor(
            self.output:view(self.output:size(1),
                             self.output:size(2),
                             self.output:size(3)*self.output:size(4),
                             self.output:size(5)))
         self.input_offset = 0
         self.output_offset = 0
         self.weight_offset = 0
         find:prepare(self, input, self.output)

   end
end
