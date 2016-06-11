local ClippedReLU, parent = torch.class('cudnn.ClippedReLU','cudnn._Pointwise')

function ClippedReLU:__init(ceiling, inplace)
    parent.__init(self)
    assert(ceiling, "No ceiling was given to ClippedReLU")
    self.ceiling = ceiling
    self.inplace = inplace or false
end

function ClippedReLU:updateOutput(input)
    if not self.mode then self.mode = 'CUDNN_ACTIVATION_CLIPPED_RELU' end
    return parent.updateOutput(self, input)
end