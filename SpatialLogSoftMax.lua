local SoftMax, parent = torch.class('cudnn.SpatialLogSoftMax', 'cudnn.SpatialSoftMax')

function SoftMax:__init(fast)
   parent.__init(self, fast)
   self.mode = 'CUDNN_SOFTMAX_MODE_CHANNEL'
   self.algorithm = 'CUDNN_SOFTMAX_LOG'
end
