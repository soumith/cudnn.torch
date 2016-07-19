local LSTM, parent = torch.class('cudnn.LSTM', 'cudnn.RNN')

function LSTM:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    self.mode = 'CUDNN_LSTM'
    self:reset()
end
