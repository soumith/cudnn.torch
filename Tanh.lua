local Tanh, parent = torch.class('cudnn.Tanh','cudnn._Pointwise')

function Tanh:__init()
   parent.__init(self)
   self.mode = 'CUDNN_ACTIVATION_TANH'
end
