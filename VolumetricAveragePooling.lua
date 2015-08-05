local VolumetricAveragePooling, parent = torch.class('cudnn.VolumetricAveragePooling', 'cudnn._Pooling3D')

function VolumetricAveragePooling:__init(kT, kW, kH, dT, dW, dH, padT, padW, padH)
   parent.__init(self, kT, kW, kH, dT, dW, dH, padT, padW, padH)
   self.mode = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
end
