local Sigmoid, parent = torch.class('cudnn.Sigmoid','cudnn._Pointwise')

function Sigmoid:__init(inplace)
   parent.__init(self, inplace)
   self.mode = 'CUDNN_ACTIVATION_SIGMOID'
end

function Sigmoid:updateOutput(input)
  if not self.mode then self.mode = 'CUDNN_ACTIVATION_SIGMOID' end
  return parent.updateOutput(self, input)
end
