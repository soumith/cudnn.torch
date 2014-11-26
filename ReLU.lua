local ReLU, parent = torch.class('cudnn.ReLU','cudnn._Pointwise')

function ReLU:__init()
   parent.__init(self)
   self.mode = 'CUDNN_ACTIVATION_RELU'
end
