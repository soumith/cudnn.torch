local SpatialMaxPooling, parent = torch.class('cudnn.SpatialMaxPooling', 'cudnn._Pooling')

function SpatialMaxPooling:__init(kW, kH, dW, dH, padW, padH)
   parent.__init(self, kW, kH, dW, dH, padW, padH)
   self.mode = 'CUDNN_POOLING_MAX'
end
