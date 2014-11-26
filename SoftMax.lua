local SoftMax, parent = torch.class('cudnn.SoftMax', 'cudnn.SpatialSoftMax')

function SoftMax:__init(fast)
   parent.__init(self, fast)
   self.mode = 'CUDNN_SOFTMAX_MODE_INSTANCE'
end
