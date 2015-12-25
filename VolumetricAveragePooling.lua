local VolumetricAveragePooling, parent = torch.class('cudnn.VolumetricAveragePooling', 'cudnn._Pooling3D')

function VolumetricAveragePooling:updateOutput(input)
   self.mode = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
   return parent.updateOutput(self, input)
end
