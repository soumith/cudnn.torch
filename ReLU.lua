local ReLU, parent = torch.class('cudnn.ReLU','cudnn._Pointwise')

function ReLU:__init(inplace)
   parent.__init(self, inplace)
   self.mode = 'CUDNN_ACTIVATION_RELU'
end

function ReLU:updateOutput(input)
  if not self.mode then self.mode = 'CUDNN_ACTIVATION_RELU' end
  return parent.updateOutput(self, input)
end
