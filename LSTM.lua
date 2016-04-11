local LSTM, parent = torch.class('cudnn.LSTM', 'cudnn.RNN')

function LSTM:__init(inputSize, hiddenSize, numLayers, batchFirst)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst)
    self.mode = 'CUDNN_LSTM'
    self:reset()
end
