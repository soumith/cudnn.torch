local SpatialAveragePooling, parent
   = torch.class('cudnn.SpatialAveragePooling', 'cudnn._Pooling')

function SpatialAveragePooling:__init(kW, kH, dW, dH, padW, padH)
   parent.__init(self, kW, kH, dW, dH, padW, padH)
   self.mode = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
end
