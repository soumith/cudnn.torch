local RNNReLU, parent = torch.class('cudnn.RNNReLU', 'cudnn.RNN')

function RNNReLU:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst, dropout)
    self.mode = 'CUDNN_RNN_RELU'
    self:reset()
end
