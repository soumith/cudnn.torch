local RNN, parent = torch.class('cudnn.RNN', 'nn.Module')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local DESCS = {'rnnDesc', 'dropoutDesc', 'wDesc', 'xDescs', 'yDescs', 'hxDesc', 'hyDesc', 'cxDesc', 'cyDesc'}

RNN.linearLayers = {
    CUDNN_LSTM = 8,
    CUDNN_GRU = 6,
    CUDNN_RNN_RELU = 2,
    CUDNN_RNN_TANH = 2
}

function RNN:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
   parent.__init(self)
   self.datatype = 'CUDNN_DATA_FLOAT'
   self.inputSize = inputSize
   self.hiddenSize = hiddenSize
   self.seqLength = 1
   self.miniBatch = 1
   self.numLayers = numLayers
   self.bidirectional = 'CUDNN_UNIDIRECTIONAL'
   self.numDirections = 1 -- set to 2 for bi-directional.
   self.inputMode = 'CUDNN_LINEAR_INPUT'
   self.mode = 'CUDNN_RNN_RELU'
   self.dropout = dropout or 0
   self.seed = 0x01234567
   self.batchFirst = batchFirst or false -- Set to true for batch x time x inputdim.
   self.rememberStates = rememberStates or false
   self.sync = true
   self.gradInput = torch.CudaTensor()
   self.output = torch.CudaTensor()
   self.weight = torch.CudaTensor()
   self.gradWeight = torch.CudaTensor()
   self.reserve = torch.CudaTensor()
   self.hiddenOutput = torch.CudaTensor()
   self.cellOutput = torch.CudaTensor()
   self.gradHiddenInput = torch.CudaTensor()
   self.gradCellInput = torch.CudaTensor()
   self:training()
   self:reset()
end

function RNN:setSync(sync)
   self.sync = sync
end

function RNN:reset(stdv)
   stdv = stdv or 1.0 / math.sqrt(self.hiddenSize)

   self:resetDropoutDescriptor()
   self:resetRNNDescriptor()
   self:resetIODescriptors()

   local weightSizePtr = ffi.new("size_t[1]")
   errcheck('cudnnGetRNNParamsSize',
            cudnn.getHandle(),
            self.rnnDesc[0],
            self.xDescs[0],
            weightSizePtr,
	    self.datatype)
   local weightSize = tonumber(weightSizePtr[0])
   local elemSize = self.weight:elementSize()
   weightSize = math.floor((weightSize + elemSize - 1) / elemSize)
   self.weight:resize(weightSize)
   self.weight:uniform(-stdv, stdv)
   self.gradWeight:resizeAs(self.weight):zero()
end

function RNN:createFilterDescriptors(count)
   return cudnn.createDescriptors(count,
                                  'cudnnFilterDescriptor_t[?]',
                                  'cudnnCreateFilterDescriptor',
                                  'cudnnDestroyFilterDescriptor')
end

function RNN:createDropoutDescriptors(count)
   return cudnn.createDescriptors(count,
                            'cudnnDropoutDescriptor_t[?]',
                            'cudnnCreateDropoutDescriptor',
                            'cudnnDestroyDropoutDescriptor')
end


function RNN:createRNNDescriptors(count)
   return cudnn.createDescriptors(count,
                            'cudnnRNNDescriptor_t[?]',
                            'cudnnCreateRNNDescriptor',
                            'cudnnDestroyRNNDescriptor')
end

function RNN:createTensorDescriptors(count)
   return cudnn.createDescriptors(count,
                            'cudnnTensorDescriptor_t[?]',
                            'cudnnCreateTensorDescriptor',
                            'cudnnDestroyTensorDescriptor')
end

function RNN:resetDropoutDescriptor()
   if not self.dropoutDesc then
      self.dropoutDesc = self:createDropoutDescriptors(1)
   end

   local dropoutStatesSizePtr = ffi.new("size_t[1]")
   errcheck('cudnnDropoutGetStatesSize',
            cudnn.getHandle(),
            dropoutStatesSizePtr)
   self.dropoutStatesSize = tonumber(dropoutStatesSizePtr[0])
   self.dropoutStates = self.dropoutStates or torch.CudaTensor()
   local nElem = ((self.dropoutStatesSize -1)/self.dropoutStates:elementSize()+1)
   self.dropoutStates:resize(nElem)

   errcheck('cudnnSetDropoutDescriptor',
            self.dropoutDesc[0],
            cudnn.getHandle(),
            self.dropout,
            self.dropoutStates:data(), self.dropoutStatesSize,
            self.seed)
end

function RNN:resetRNNDescriptor()
   if not self.rnnDesc then
      self.rnnDesc = self:createRNNDescriptors(1)
   end
   errcheck('cudnnSetRNNDescriptor',
            self.rnnDesc[0],
            self.hiddenSize,
            self.numLayers,
            self.dropoutDesc[0],
            self.inputMode,
            self.bidirectional,
            self.mode,
            self.datatype)
end

function RNN:resetWeightDescriptor()
   self.wDesc =  cudnn.setFilterDescriptor(
      { dataType = self.datatype,
        filterDimA = {self.weight:size(1), 1, 1}
      }
   )
end

function RNN:resetIODescriptors()
   self.xDescs = self:createTensorDescriptors(self.seqLength)
   self.yDescs = self:createTensorDescriptors(self.seqLength)

   for i = 0, self.seqLength - 1 do
      local dim = torch.IntTensor({ self.miniBatch,self.inputSize, 1})
      local stride = torch.IntTensor({dim[3] * dim[2], dim[3],1})
      errcheck('cudnnSetTensorNdDescriptor',
               self.xDescs[i],
               self.datatype,
               3,
               dim:data(),
               stride:data())

      local dim = torch.IntTensor({self.miniBatch, self.hiddenSize * self.numDirections, 1})
      local stride = torch.IntTensor({dim[3] * dim[2], dim[3],1})
      errcheck('cudnnSetTensorNdDescriptor',
               self.yDescs[i],
               self.datatype,
               3,
               dim:data(),
               stride:data())
   end
end

function RNN:resetHiddenDescriptors()
   self.hxDesc = self:createTensorDescriptors(1)
   self.hyDesc = self:createTensorDescriptors(1)
   local dim = torch.IntTensor({self.numLayers*self.numDirections, self.miniBatch, self.hiddenSize })
   local stride = torch.IntTensor({dim[3] * dim[2], dim[3],1})

   errcheck('cudnnSetTensorNdDescriptor',
            self.hxDesc[0],
            self.datatype,
            3,
            dim:data(),
            stride:data())
   errcheck('cudnnSetTensorNdDescriptor',
            self.hyDesc[0],
            self.datatype,
            3,
            dim:data(),
            stride:data())
end

function RNN:resetCellDescriptors()
   self.cxDesc = self:createTensorDescriptors(1)
   self.cyDesc = self:createTensorDescriptors(1)
   local dim = torch.IntTensor({self.numLayers*self.numDirections, self.miniBatch, self.hiddenSize })
   local stride = torch.IntTensor({dim[3] * dim[2], dim[3],1})

   errcheck('cudnnSetTensorNdDescriptor',
            self.cxDesc[0],
            self.datatype,
            3,
            dim:data(),
            stride:data())
   errcheck('cudnnSetTensorNdDescriptor',
            self.cyDesc[0],
            self.datatype,
            3,
            dim:data(),
            stride:data())
end

function RNN:makeContiguous(input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function RNN:resizeOutput(tensor)
    return tensor:resize(self.seqLength, self.miniBatch, self.hiddenSize * self.numDirections)
end

function RNN:resizeHidden(tensor)
    return tensor:resize(self.numLayers * self.numDirections, self.miniBatch, self.hiddenSize)
end

function RNN:resetStates()
   if self.hiddenInput then
      self.hiddenInput = nil
   end
   if self.cellInput then
      self.cellInput = nil
   end
   if self.gradHiddenOutput then
      self.gradHiddenOutput = nil
   end
   if self.gradCellOutput then
      self.gradCellOutput = nil
   end
end




function RNN:updateOutput(input)
    if (self.batchFirst) then
        input = input:transpose(1, 2)
    end
   assert(input:dim() == 3, 'input must have 3 dimensions: seqLength, miniBatch, inputSize')
   assert(self.dropout == 0 or cudnn.version >= 5103, 'dropout supported only in cudnn v5.1 and above')
   -- Decide which descriptors/tensors need to be updated.
   local resetRNN = not self.dropoutDesc or not self.rnnDesc
   local resetIO = not self.xDescs or not self.yDescs
   local resetHC = not self.hxDesc or not self.hyDesc or not self.cxDesc or not self.cyDesc
   local resetWeight = not self.wDesc

   if input:size(1) ~= self.seqLength then
      self.seqLength = input:size(1)
      resetRNN = true
      resetIO = true
   end

   if input:size(2) ~= self.miniBatch then
      self.miniBatch = input:size(2)
      resetIO = true
      resetHC = true
   end

   assert(input:size(3) == self.inputSize, 'Incorrect input size!')


   -- Update descriptors/tensors
   if resetRNN then
      if not self.dropoutDesc then self:resetDropoutDescriptor() end
      self:resetRNNDescriptor()
   end
   if resetIO then
      self:resetIODescriptors(input)
   end
   if resetHC then
      self:resetHiddenDescriptors()
      self:resetCellDescriptors()
   end
   if resetWeight then
      self:resetWeightDescriptor()
   end

   local x = self:makeContiguous(input)
   local oSize = torch.LongStorage({self.seqLength, self.miniBatch, self.hiddenSize * self.numDirections})
   local oStride = torch.LongStorage({self.miniBatch * self.hiddenSize * self.numDirections, self.hiddenSize * self.numDirections, 1})
   self.output:resize(oSize, oStride)
   local y = self.output
   local w = self.weight

   -- Optionally use hiddenInput/cellInput parameters
   if self.rememberStates then
        if self.hiddenOutput:nDimension() == 3 and self.hiddenOutput:size(1) == self.numLayers * self.numDirections and 
           self.hiddenOutput:size(2) == self.miniBatch and self.hiddenOutput:size(3) == self.hiddenSize then
	       self.hiddenInput = self.hiddenOutput:clone()
	       if self.cellOutput and self.cellOutput:isSameSizeAs(self.hiddenOutput) then
 	           self.cellInput = self.cellOutput:clone()
               end
        else
	   self.hiddenInput = nil
           self.cellInput = nil
        end     
   end
   local hx = self.hiddenInput
   local cx = self.cellInput
   local hy = self:resizeHidden(self.hiddenOutput):zero()
   local cy = self:resizeHidden(self.cellOutput):zero()

   if hx then
      assert(hx:dim() == 3, 'hiddenInput must have 3 dimensions: numLayers, miniBatch, hiddenSize')
      assert(hx:size(1) == self.numLayers * self.numDirections, 'hiddenInput has incorrect number of layers!')
      assert(hx:size(2) == self.miniBatch, 'hiddenInput has incorrect number of minibathes!')
      assert(hx:size(3) == self.hiddenSize, 'hiddenInput has incorrect size!')
      assert(hx:isContiguous(), 'hiddenInput must be contiguous!') end

   if cx then
      assert(cx:dim() == 3, 'cellInput must have 3 dimensions: numLayers, miniBatch, hiddenSize')
      assert(cx:size(1) == self.numLayers * self.numDirections, 'cellInput has incorrect number of layers!')
      assert(cx:size(2) == self.miniBatch, 'cellInput has incorrect number of minibathes!')
      assert(cx:size(3) == self.hiddenSize, 'cellInput has incorrect size!')
      assert(cx:isContiguous(), 'cellInput must be contiguous!')
   end

   local workspaceSizePtr = ffi.new("size_t[1]")
   errcheck('cudnnGetRNNWorkspaceSize',
            cudnn.getHandle(),
            self.rnnDesc[0],
	    self.seqLength,
            self.xDescs,
            workspaceSizePtr)
   local workspaceSize = tonumber(workspaceSizePtr[0])
   cudnn.setSharedWorkspaceSize(workspaceSize, true)
   local wsPtr, wsSize = cudnn.getSharedWorkspace()

   if self.train then
      local reserveSizePtr = ffi.new("size_t[1]")
      errcheck('cudnnGetRNNTrainingReserveSize',
               cudnn.getHandle(),
               self.rnnDesc[0],
	       self.seqLength,
               self.xDescs,
               reserveSizePtr)
      local reserveSize = tonumber(reserveSizePtr[0])
      local elemSize = self.reserve:elementSize()
      reserveSize = math.floor((reserveSize + elemSize - 1) / elemSize)
      self.reserve:resize(reserveSize)
      errcheck('cudnnRNNForwardTraining',
               cudnn.getHandle(),
               self.rnnDesc[0],
	       self.seqLength,
               self.xDescs, x:data(),
               self.hxDesc[0], hx and hx:data() or nil,
               self.cxDesc[0], cx and cx:data() or nil,
               self.wDesc[0], w:data(),
               self.yDescs, y:data(),
               self.hyDesc[0], hy:data(),
               self.cyDesc[0], cy:data(),
               wsPtr,
	       wsSize,
               self.reserve:data(), self.reserve:size(1) * self.reserve:elementSize())
   else
      errcheck('cudnnRNNForwardInference',
               cudnn.getHandle(),
               self.rnnDesc[0],
	       self.seqLength,
               self.xDescs, x:data(),
               self.hxDesc[0], hx and hx:data() or nil,
               self.cxDesc[0], cx and cx:data() or nil,
               self.wDesc[0], w:data(),
               self.yDescs, y:data(),
               self.hyDesc[0], hy:data(),
               self.cyDesc[0], cy:data(),
               wsPtr,
	       wsSize)
   end
   if self.sync then cutorch.synchronize() end
   if (self.batchFirst) then
      self.output = self.output:transpose(1, 2)
   end
   return self.output
end

function RNN:updateGradInput(input, gradOutput)
    if (self.batchFirst) then
        input = input:transpose(1, 2)
        gradOutput = gradOutput:transpose(1, 2)
        self.output = self.output:transpose(1, 2)
    end
   assert(self.dropout == 0 or cudnn.version >= 5103, 'dropout supported only in cudnn v 5.1 and above')
   assert(input:dim() == 3, 'input should have 3 dimensions: seqLength, miniBatch, inputSize')
   assert(input:size(1) == self.seqLength, 'input has incorrect sequence length!')
   assert(input:size(2) == self.miniBatch, 'input has incorrect minibatch size!')
   assert(input:size(3) == self.inputSize, 'input has incorrect size!')

   assert(gradOutput:isSameSizeAs(self.output), 'gradOutput has incorrect size!')
   assert(self.train, 'updateGradInput can only be called when training!')

   local x, dy = self:makeContiguous(input, gradOutput)
   local y = self.output
   local w = self.weight
   local dx = self.gradInput:resizeAs(input)
   local hx = self.hiddenInput
   local cx = self.cellInput
   local dhy = self.gradHiddenOutput
   local dcy = self.gradCellOutput
   local dhx = self:resizeHidden(self.gradHiddenInput):zero()
   local dcx = self:resizeHidden(self.gradCellInput):zero()


   if hx then
      assert(hx:dim() == 3, 'hiddenInput must have 3 dimensions: numLayers, miniBatch, hiddenSize')
      assert(hx:size(1) == self.numLayers * self.numDirections, 'hiddenInput has incorrect number of layers!')
      assert(hx:size(2) == self.miniBatch, 'hiddenInput has incorrect minibatch size!')
      assert(hx:size(3) == self.hiddenSize, 'hiddenInput has incorrect size!')
      assert(hx:isContiguous(), 'hiddenInput must be contiguous!')
   end

   if cx then
      assert(cx:dim() == 3, 'cellInput must have 3 dimensions: numLayers, miniBatch, hiddenSize')
      assert(cx:size(1) == self.numLayers * self.numDirections, 'cellInput has incorrect number of layers!')
      assert(cx:size(2) == self.miniBatch, 'cellInput has incorrect minibatch size!')
      assert(cx:size(3) == self.hiddenSize, 'cellInput has incorrect size!')
      assert(cx:isContiguous(), 'cellInput must be contiguous!')
   end

   if dhy then
      assert(dhy:dim() == 3, 'gradHiddenOutput must have 3 dimensions: ' ..
                             'numLayers, miniBatch, hiddenSize')
      assert(dhy:size(1) == self.numLayers * self.numDirections, 'gradHiddenOutput has incorrect number of layers!')
      assert(dhy:size(2) == self.miniBatch, 'gradHiddenOutput has incorrect minibatch size!')
      assert(dhy:size(3) == self.hiddenSize, 'gradHiddenOutput has incorrect size!')
      assert(dhy:isContiguous(), 'gradHiddenOutput must be contiguous!')
   end

   if dcy then
      assert(dcy:dim() == 3, 'gradCellOutput must have 3 dimensions: ' ..
                             'numLayers, miniBatch, hiddenSize')
      assert(dcy:size(1) == self.numLayers * self.numDirections, 'gradCellOutput has incorrect number of layers!')
      assert(dcy:size(2) == self.miniBatch, 'gradCellOutput has incorrect minibatch size!')
      assert(dcy:size(3) == self.hiddenSize, 'gradCellOutput has incorrect size!')
      assert(dcy:isContiguous(), 'gradCellOutput must be contiguous!')
   end
   local workspaceSizePtr = ffi.new("size_t[1]")
   errcheck('cudnnGetRNNWorkspaceSize',
            cudnn.getHandle(),
            self.rnnDesc[0],
	    self.seqLength,
            self.xDescs,
            workspaceSizePtr)
   local workspaceSize = tonumber(workspaceSizePtr[0])
   cudnn.setSharedWorkspaceSize(workspaceSize, true)
   local wsPtr, wsSize = cudnn.getSharedWorkspace()

   errcheck('cudnnRNNBackwardData',
            cudnn.getHandle(),
            self.rnnDesc[0],
	    self.seqLength,
            self.yDescs, y:data(),
            self.yDescs, dy:data(),
            self.hyDesc[0], dhy and dhy:data() or nil,
            self.cyDesc[0], dcy and dcy:data() or nil,
            self.wDesc[0], w:data(),
            self.hxDesc[0], hx and hx:data() or nil,
            self.cxDesc[0], cx and cx:data() or nil,
            self.xDescs, dx:data(),
            self.hxDesc[0], dhx:data(),
            self.cxDesc[0], dcx:data(),
	    wsPtr, wsSize,
            self.reserve:data(), self.reserve:size(1) * self.reserve:elementSize())
    if self.sync then cutorch.synchronize() end
    if (self.batchFirst) then
        self.gradInput = self.gradInput:transpose(1, 2)
        self.output = self.output:transpose(1, 2)
    end
   return self.gradInput
end

function RNN:accGradParameters(input, gradOutput, scale)
    if (self.batchFirst) then
        input = input:transpose(1, 2)
        gradOutput = gradOutput:transpose(1, 2)
        self.output = self.output:transpose(1, 2)
    end
   scale = scale or 1
   if scale == 0 then return end
   assert(self.dropout == 0 or cudnn.version >= 5103, 'dropout supported only in cudnn 5.1 and above')
   assert(input:dim() == 3, 'input should have 3 dimensions: seqLength, miniBatch, inputSize')
   assert(input:size(1) == self.seqLength, 'input has incorrect sequence length!')
   assert(input:size(2) == self.miniBatch, 'input has incorrect minibatch size!')
   assert(input:size(3) == self.inputSize, 'input has incorrect size!')

   assert(gradOutput:isSameSizeAs(self.output), 'gradOutput has incorrect size!')
   assert(self.train, 'accGradParameters can only be called when training!')

   local x, dy = self:makeContiguous(input, gradOutput)
   local hx = self.hiddenInput
   local y = self.output
   local dw = self.gradWeight

   if hx then
      assert(hx:dim() == 3, 'hiddenInput must have 3 dimensions: numLayers, miniBatch, hiddenSize')
      assert(hx:size(1) == self.numLayers * self.numDirections, 'hiddenInput has incorrect number of layers!')
      assert(hx:size(2) == self.miniBatch, 'hiddenInput has incorrect minibatch size!')
      assert(hx:size(3) == self.hiddenSize, 'hiddenInput has incorrect size!')
      assert(hx:isContiguous(), 'hiddenInput must be contiguous!')
   end

   -- cudnnRNNBackwardWeights doesn't accept a scale parameter so instead
   -- scale before and after.
   -- TODO: How much does this impact accuracy?
   --       Use a secondary buffer instead?
   if scale ~= 1 then
      local scaleTensor = torch.Tensor({1 / scale})
      errcheck('cudnnScaleTensor',
               cudnn.getHandle(),
               self.wDesc[0],
               self.dw:data(),
               scaleTensor:data())
   end
   local workspaceSizePtr = ffi.new("size_t[1]")
   errcheck('cudnnGetRNNWorkspaceSize',
            cudnn.getHandle(),
            self.rnnDesc[0],
	    self.seqLength,
            self.xDescs,
            workspaceSizePtr)
   local workspaceSize = tonumber(workspaceSizePtr[0])
   cudnn.setSharedWorkspaceSize(workspaceSize, true)
   local wsPtr, wsSize = cudnn.getSharedWorkspace()

   errcheck('cudnnRNNBackwardWeights',
            cudnn.getHandle(),
            self.rnnDesc[0],
	    self.seqLength,
            self.xDescs, x:data(),
            self.hxDesc[0], hx and hx:data() or nil,
            self.yDescs, y:data(),
	    wsPtr, wsSize,
            self.wDesc[0], dw:data(),
            self.reserve:data(), self.reserve:size(1) * self.reserve:elementSize())

   if scale ~= 1 then
      local scaleTensor = torch.Tensor({scale})
      errcheck('cudnnScaleTensor',
               cudnn.getHandle(),
               self.wDesc[0],
               self.dw:data(),
               scaleTensor:data())
   end

    if (self.batchFirst) then
        gradOutput = gradOutput:transpose(1, 2)
        self.output = self.output:transpose(1, 2)
    end
end

local function numberOfLinearLayers(self)
    return self.linearLayers[self.mode]
end

local function numberOfLayers(self)
    if self.bidirectional == 'CUDNN_BIDIRECTIONAL' then
        assert(self.numDirections == 2)
        return 2 * self.numLayers
    else
        return self.numLayers
    end
end

-- Function gets either the matrix or bias param x on cuDNN method given, at each layer and linear layerId.
local function retrieveLinearParams(self, cuDNNMethod)
    if not self.wDesc then
        self:resetWeightDescriptor()
    end
    local linearParams = {}
    local numberOfLinearLayers = numberOfLinearLayers(self)
    local numLayers = numberOfLayers(self)
    for layer = 0, numLayers - 1 do
        local layerInfo = {}
        for layerId = 0, numberOfLinearLayers - 1 do
            local linLayerMatDesc = self:createFilterDescriptors(1)
            local matrixPointer = ffi.new("float*[1]")
            errcheck(cuDNNMethod,
                cudnn.getHandle(),
                self.rnnDesc[0],
                layer,
                self.xDescs[0],
                self.wDesc[0],
                self.weight:data(),
                layerId,
                linLayerMatDesc[0],
                ffi.cast("void**", matrixPointer))

            local dataType = ffi.new("cudnnDataType_t[1]")
            local format = ffi.new("cudnnTensorFormat_t[1]")
            local nbDims = torch.IntTensor(1)

            local minDim = 3
            local filterDimA = torch.ones(minDim):int()
            errcheck('cudnnGetFilterNdDescriptor',
                linLayerMatDesc[0],
                minDim,
                dataType,
                format,
                nbDims:data(),
                filterDimA:data())

            local offset = matrixPointer[0] - self.weight:data()
            local params = torch.CudaTensor(self.weight:storage(), offset + self.weight:storageOffset(), filterDimA:prod())
            table.insert(layerInfo, params)
        end
        table.insert(linearParams, layerInfo)
    end
    return linearParams
end

function RNN:weights()
    return retrieveLinearParams(self, 'cudnnGetRNNLinLayerMatrixParams')
end

function RNN:biases()
    return retrieveLinearParams(self, 'cudnnGetRNNLinLayerBiasParams')
end

function RNN:clearDesc()
   for _, desc in pairs(DESCS) do
      self[desc] = nil
   end
end

function RNN:write(f)
   local pushDescs = {}
   for _, desc in pairs(DESCS) do
      pushDescs[desc] = self[desc]
   end

   self:clearDesc()

   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)

   for desc, v in pairs(pushDescs) do
      self[desc] = v
   end
end

function RNN:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_input', '_gradOutput', 'reserve', 'dropoutStates')
   return parent.clearState(self)
end
