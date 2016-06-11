local ClippedReLU, parent = torch.class('cudnn.ClippedReLU','cudnn._Pointwise')

function ClippedReLU:__init(inplace, ceiling)
    parent.__init(self)
    self.inplace = inplace
    assert(ceiling, "No ceiling was given to ClippedReLU")
    self.ceiling = ceiling
end

function ClippedReLU:updateOutput(input)
    if not self.mode then self.mode = 'CUDNN_ACTIVATION_CLIPPED_RELU' end
    return parent.updateOutput(self, input)
end