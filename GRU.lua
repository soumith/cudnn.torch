local GRU, parent = torch.class('cudnn.GRU', 'cudnn.RNN')

function GRU:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    self.mode = 'CUDNN_GRU'
    self:reset()
end
