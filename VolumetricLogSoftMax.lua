local SoftMax, parent = torch.class('cudnn.VolumetricLogSoftMax', 'cudnn.VolumetricSoftMax')

function SoftMax:__init(fast)
   parent.__init(self, fast)
   self.ssm.mode = 'CUDNN_SOFTMAX_MODE_CHANNEL'
   self.ssm.algorithm = 'CUDNN_SOFTMAX_LOG'
end
