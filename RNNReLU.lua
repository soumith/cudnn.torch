local RNNReLU, parent = torch.class('cudnn.RNNReLU', 'cudnn.RNN')

function RNNReLU:__init(inputSize, hiddenSize, numLayers, batchFirst)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst)
    self.mode = 'CUDNN_RNN_RELU'
    self:reset()
end
