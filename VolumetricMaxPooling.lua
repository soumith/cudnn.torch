local VolumetricMaxPooling, parent = torch.class('cudnn.VolumetricMaxPooling',
                                                 'cudnn._Pooling3D')

function VolumetricMaxPooling:updateOutput(input)
   self.mode = 'CUDNN_POOLING_MAX'
   return parent.updateOutput(self, input)
end
