local SoftMax, parent = torch.class('cudnn.SoftMax', 'cudnn.SpatialSoftMax')

function SoftMax:updateOutput(input)
   self.mode = 'CUDNN_SOFTMAX_MODE_INSTANCE'
   return parent.updateOutput(self, input)
end
