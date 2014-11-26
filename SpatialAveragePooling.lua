local SpatialAveragePooling, parent
   = torch.class('cudnn.SpatialAveragePooling', 'cudnn._Pooling')

function SpatialAveragePooling:__init(kW, kH, dW, dH)
   parent.__init(self, kW, kH, dW, dH)
   self.mode = 'CUDNN_POOLING_AVERAGE'
end
