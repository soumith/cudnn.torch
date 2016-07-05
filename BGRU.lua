local BGRU, parent = torch.class('cudnn.BGRU', 'cudnn.RNN')

function BGRU:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout)
    parent.__init(self, inputSize, hiddenSize, numLayers, batchFirst, dropout)
    self.bidirectional = 'CUDNN_BIDIRECTIONAL'
    self.mode = 'CUDNN_GRU'
    self.numDirections = 2
    self:reset()
end
