local SoftMax, parent = torch.class('cudnn.LogSoftMax', 'cudnn.SpatialSoftMax')

function SoftMax:__init(fast)
   parent.__init(self, fast)
   self.mode = 'CUDNN_SOFTMAX_MODE_INSTANCE'
   self.algorithm = 'CUDNN_SOFTMAX_LOG'
end
