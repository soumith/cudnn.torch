local SpatialDilatedConvolution, parent =
    torch.class('cudnn.SpatialDilatedConvolution', 'cudnn.SpatialConvolution')
local ffi = require 'ffi'
local find = require 'cudnn.find'

function SpatialDilatedConvolution:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH, dilationW, dilationH, groups)
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, groups)--, dilationW, dilationH)
    self.dilationW = dilationW
    self.dilationH = dilationH
end

function SpatialDilatedConvolution:createIODescriptors(input)
   local batch = true
   if input:dim() == 3 then
      input = input:view(1, input:size(1), input:size(2), input:size(3))
      batch = false
   end
   if parent.checkInputChanged(self, input) then
        -- create input descriptor
        local input_slice = input:narrow(2,1,self.nInputPlane/self.groups)
        self.iDesc = cudnn.toDescriptor(input_slice)
        -- create conv descriptor
        self.padH, self.padW = self.padH or 0, self.padW or 0
        -- those needed to calculate hash
        self.pad = {self.padH, self.padW}
        self.stride = {self.dH, self.dW}
	local t_dataType = cudnn.configmap(torch.type(self.weight))
	--fallback to fp32 math if half type, fp16 dilated convs not fully implmented in cuDNN 6.0.2
	if( t_dataType == 'CUDNN_DATA_HALF') then t_dataType = 'CUDNN_DATA_FLOAT' end
	self.convDescData = {
           padA = self.pad,
           filterStrideA = self.stride,
           dilationA = {self.dilationH, self.dilationW},
           dataType = t_dataType
        }
        self.convDesc = cudnn.setConvolutionDescriptor(self.convDescData)

        -- get output shape, resize output
        local oSize = torch.IntTensor(4)
        cudnn.errcheck('cudnnGetConvolutionNdForwardOutputDim',
                 self.convDesc[0], self.iDesc[0],
                 self.weightDesc[0], 4, oSize:data())
        oSize[2] = oSize[2] * self.groups
        self.output:resize(oSize:long():storage())
        self.oSize = self.output:size()

        local output_slice = self.output:narrow(2,1,self.nOutputPlane/self.groups)
        -- create descriptor for output
        self.oDesc = cudnn.toDescriptor(output_slice)
        self.oDescForBias = cudnn.toDescriptor(self.output)

        find:prepare(self, input_slice, output_slice)

        -- create offsets for groups
        local iH, iW = input:size(3), input:size(4)
        local kH, kW = self.kH, self.kW
        local oH, oW = oSize[3], oSize[4]
        self.input_offset = self.nInputPlane / self.groups * iH * iW
        self.output_offset = self.nOutputPlane / self.groups * oH * oW
        self.weight_offset = self.nInputPlane / self.groups * self.nOutputPlane / self.groups * kH * kW

        if not batch then
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
        end

   end
   return self
end
