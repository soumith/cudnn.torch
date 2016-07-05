local RNNTanh, parent = torch.class('cudnn.RNNTanh', 'cudnn.RNN')

function RNNTanh:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst, dropout)
    self.mode = 'CUDNN_RNN_TANH'
    self:reset()
end
