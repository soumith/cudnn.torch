local VolumetricMaxPooling, parent = torch.class('cudnn.VolumetricMaxPooling', 'cudnn._Pooling3D')

function VolumetricMaxPooling:__init(kT, kW, kH, dT, dW, dH, padT, padW, padH)
   parent.__init(self, kT, kW, kH, dT, dW, dH, padT, padW, padH)
   self.mode = 'CUDNN_POOLING_MAX'
end
