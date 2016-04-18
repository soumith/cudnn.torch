local GRU, parent = torch.class('cudnn.GRU', 'cudnn.RNN')

function GRU:__init(inputSize, hiddenSize, numLayers, batchFirst)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst)
    self.mode = 'CUDNN_GRU'
    self:reset()
end
