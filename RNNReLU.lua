local RNNReLU, parent = torch.class('cudnn.RNNReLU', 'cudnn.RNN')

function RNNReLU:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    self.mode = 'CUDNN_RNN_RELU'
    self:reset()
end
