local SoftMax, parent = torch.class('cudnn.LogSoftMax', 'cudnn.SpatialSoftMax')

function SoftMax:updateOutput(input)
   self.mode = 'CUDNN_SOFTMAX_MODE_INSTANCE'
   self.algorithm = 'CUDNN_SOFTMAX_LOG'
   return parent.updateOutput(self, input)
end
