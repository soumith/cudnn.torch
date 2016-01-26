local Tanh, parent = torch.class('cudnn.Tanh','cudnn._Pointwise')

function Tanh:updateOutput(input)
  if not self.mode then self.mode = 'CUDNN_ACTIVATION_TANH' end
  return parent.updateOutput(self, input)
end
