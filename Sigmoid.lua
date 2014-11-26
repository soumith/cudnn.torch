local Sigmoid, parent = torch.class('cudnn.Sigmoid','cudnn._Pointwise')

function Sigmoid:__init()
   parent.__init(self)
   self.mode = 'CUDNN_ACTIVATION_SIGMOID'
end
