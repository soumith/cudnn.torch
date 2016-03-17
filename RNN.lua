local RNN, parent = torch.class('cudnn.RNN', 'nn.Module')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function RNN:__init(inputSize, hiddenSize, numLayers)
    parent.__init(self)

    self.datatype = 'CUDNN_DATA_FLOAT'
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.numLayers = numLayers
    self.bidirectional = 'CUDNN_UNIDIRECTIONAL'
    self.inputMode = 'CUDNN_LINEAR_INPUT'
    self.mode = 'CUDNN_RNN_RELU'
    self.dropout = 0
    self.seed = 0x01234567

    self.gradInput = torch.CudaTensor()
    self.output = torch.CudaTensor()
    self.weight = torch.CudaTensor()
    self.gradWeight = torch.CudaTensor()
    self.reserve = torch.CudaTensor(1)
    self.hiddenOutput = torch.CudaTensor()
    self.cellOutput = torch.CudaTensor()
    self.gradHiddenInput = torch.CudaTensor()
    self.gradCellInput = torch.CudaTensor()

    self:training()
end

local function createDescriptors(count, descs_type, create_func, destroy_func)
    local ds = ffi.new(descs_type, count)
    for i = 0, count - 1 do
        errcheck(create_func, ds + i)
    end
    local function destroyDescriptors(ds)
        for i = 0, count - 1 do
            errcheck(destroy_func, ds[i])
        end
    end
    ffi.gc(ds, destroyDescriptors)
    return ds
end

local function createDropoutDescriptors(count)
    return createDescriptors(count,
                             'cudnnDropoutDescriptor_t[?]',
                             'cudnnCreateDropoutDescriptor',
                             'cudnnDestroyDropoutDescriptor')
end

local function createFilterDescriptors(count)
    return createDescriptors(count,
                             'cudnnFilterDescriptor_t[?]',
                             'cudnnCreateFilterDescriptor',
                             'cudnnDestroyFilterDescriptor')
end

local function createRNNDescriptors(count)
    return createDescriptors(count,
                             'cudnnRNNDescriptor_t[?]',
                             'cudnnCreateRNNDescriptor',
                             'cudnnDestroyRNNDescriptor')
end

local function createTensorDescriptors(count)
    return createDescriptors(count,
                             'cudnnTensorDescriptor_t[?]',
                             'cudnnCreateTensorDescriptor',
                             'cudnnDestroyTensorDescriptor')
end

function RNN:resetDropoutDescriptor()
    if not self.dropoutDesc then
        self.dropoutDesc = createDropoutDescriptors(1)
    end

    self.dropoutStatesSize = torch.LongTensor(1)
    errcheck('cudnnDropoutGetStatesSize',
             cudnn.getHandle(),
             self.dropoutStatesSize:data())
    self.dropoutStates = torch.CudaTensor(self.dropoutStatesSize[1])

    errcheck('cudnnSetDropoutDescriptor',
             self.dropoutDesc[0],
             cudnn.getHandle(),
             self.dropout,
             self.dropoutStates:data(), self.dropoutStatesSize[1],
             self.seed)
end

function RNN:resetRNNDescriptor()
    if not self.rnnDesc then
        self.rnnDesc = createRNNDescriptors(1)
    end

    errcheck('cudnnSetRNNDescriptor',
             self.rnnDesc[0],
             self.hiddenSize,
             self.seqLength,
             self.numLayers,
             self.dropoutDesc[0],
             self.inputMode,
             self.bidirectional,
             self.mode,
             self.datatype)
end

function RNN:resetWeightDescriptors()
    if not self.wDesc then
        self.wDesc = createFilterDescriptors(1)
    end

    local weightSize = torch.LongTensor(1)
    errcheck('cudnnGetRNNParamsSize',
             cudnn.getHandle(),
             self.rnnDesc[0],
             self.xDescs,
             weightSize:data())
    local dim = torch.IntTensor({weightSize[1] / 4, 1, 1}) -- sizeof(float)

    errcheck('cudnnSetFilterNdDescriptor',
             self.wDesc[0],
             self.datatype,
             'CUDNN_TENSOR_NCHW',
             3,
             dim:data())
end

function RNN:resetIODescriptors(input)
    self.xDescs = createTensorDescriptors(input:size(1))
    self.yDescs = createTensorDescriptors(self.output:size(1))

    for i = 0, self.seqLength - 1 do
        local dim = torch.IntTensor({input:size(3), input:size(2), input:size(1)})
        local stride = torch.IntTensor({1, dim[1], dim[1] * dim[2]})
        errcheck('cudnnSetTensorNdDescriptor',
                 self.xDescs[i],
                 self.datatype,
                 3,
                 dim:data(),
                 stride:data())

        local dim = torch.IntTensor({self.output:size(3), self.output:size(2), self.output:size(1)})
        local stride = torch.IntTensor({1, dim[1], dim[1] * dim[2]})
        errcheck('cudnnSetTensorNdDescriptor',
                 self.yDescs[i],
                 self.datatype,
                 3,
                 dim:data(),
                 stride:data())
    end
end

function RNN:resetHiddenDescriptors()
    self.hxDesc = createTensorDescriptors(1)
    self.hyDesc = createTensorDescriptors(1)

    local dim = torch.IntTensor({self.hiddenSize, self.miniBatch, self.numLayers})
    local stride = torch.IntTensor({1, dim[1], dim[1] * dim[2]})

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
    self.cxDesc = createTensorDescriptors(1)
    self.cyDesc = createTensorDescriptors(1)

    local dim = torch.IntTensor({self.hiddenSize, self.miniBatch, self.numLayers})
    local stride = torch.IntTensor({1, dim[1], dim[1] * dim[2]})

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

local function makeContiguous(self, input, gradOutput)
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

function RNN:updateOutput(input)

    assert(input:dim() == 3, 'Input should have three dimensions: (seqLength, miniBatch, inputSize)')

    -- Decide which descriptors/tensors need to be updated.
    local resetRNN = not DropoutDesc or not RNNDesc
    local resetIO = not xDescs or not yDescs
    local resetHC = not self.hxDesc or not self.hyDesc or
                    not self.cxDesc or not self.cyDesc
    local resetWeight = not wDesc

    if input:size(1) ~= self.seqLength then
        self.seqLength = input:size(1)
        resetRNN = true
        resetIO = true
    end

    if input:size(2) ~= self.miniBatch then
        self.miniBatch = input:size(2)
        resetRNN = true
        resetIO = true
        resetHC = true
    end

    assert(resetWeight or input:size(3) == self.inputSize, 'Input size has changed! clearState() must be called to resize weights.')

    -- Update descriptors/tensors
    if resetRNN then
        self:resetDropoutDescriptor()
        self:resetRNNDescriptor()
    end

    local x = makeContiguous(self, input)
    local y = self.output:resize(self.seqLength, self.miniBatch, self.hiddenSize)
    if resetIO then
        self:resetIODescriptors(input)
    end

    -- Optionally use hiddenInput/cellInput parameters
    local hx = self.hiddenInput
    if hx then
      assert(hx:dim() == 3, 'Hidden input must have 3 dimensions: (numLayers, miniBatch, hiddenSize)')
      assert(hx:size(1) == self.numLayers, 'Hidden input has incorrect number of layers!')
      assert(hx:size(2) == self.miniBatch, 'Hidden input has incorrect number of minibathes!')
      assert(hx:size(3) == self.hiddenSize, 'Hidden input has incorrect size!')
    end

    local cx = self.cellInput
    if cx then
      assert(cx:dim() == 3, 'Cell input must have 3 dimensions: (numLayers, miniBatch, hiddenSize)')
      assert(cx:size(1) == self.numLayers, 'Cell input has incorrect number of layers!')
      assert(cx:size(2) == self.miniBatch, 'Cell input has incorrect number of minibathes!')
      assert(cx:size(3) == self.hiddenSize, 'Cell input has incorrect size!')
    end

    local hy = self.hiddenOutput:resize(self.numLayers, self.miniBatch, self.hiddenSize)
    local cy = self.cellOutput:resize(self.numLayers, self.miniBatch, self.hiddenSize)

    if resetHC then
        self:resetHiddenDescriptors()
        self:resetCellDescriptors()
    end

    local w = self.weight
    if resetWeight then
        local weightSize = torch.LongTensor(1)
        errcheck('cudnnGetRNNParamsSize',
                 cudnn.getHandle(),
                 self.rnnDesc[0],
                 self.xDescs,
                 weightSize:data())
        weightSize[1] = (weightSize[1] + 3) / 4 -- sizeof(float)
        self.weight:resize(weightSize[1] / 4)
        self.gradWeight:resizeAs(self.weight):zero()
        self:resetWeightDescriptors()
    end

    self.workspace = cudnn.getSharedWorkspace()
    local workspaceSize = torch.LongTensor(1)
    errcheck('cudnnGetRNNWorkspaceSize',
             cudnn.getHandle(),
             self.rnnDesc[0],
             self.xDescs,
             workspaceSize:data())
    workspaceSize[1] = (workspaceSize[1] + 3) / 4 -- sizeof(float)
    if self.workspace:size(1) < workspaceSize[1] then
        self.workspace:resize(workspaceSize[1])
    end

    if self.train then
        local reserveSize = torch.LongTensor(1)
        errcheck('cudnnGetRNNTrainingReserveSize',
                 cudnn.getHandle(),
                 self.rnnDesc[0],
                 self.xDescs,
                 reserveSize:data())
        reserveSize[1] = (reserveSize[1] + 3) / 4 -- sizeof(float)
        if self.reserve:size(1) < reserveSize[1] then
            self.reserve:resize(reserveSize[1])
        end

        errcheck('cudnnRNNForwardTraining',
                 cudnn.getHandle(),
                 self.rnnDesc[0],
                 self.xDescs, x:data(),
                 self.hxDesc[0], hx and hx:data() or nil,
                 self.cxDesc[0], cx and cx:data() or nil,
                 self.wDesc[0], w:data(),
                 self.yDescs, y:data(),
                 self.hyDesc[0], hy:data(),
                 self.cyDesc[0], cy:data(),
                 self.workspace:data(), self.workspace:size(1) * 4, -- sizeof(float)
                 self.reserve:data(), self.reserve:size(1) * 4) -- sizeof(float)
    else
        errcheck('cudnnRNNForwardInference',
                 cudnn.getHandle(),
                 self.rnnDesc[0],
                 self.xDescs, x:data(),
                 self.hxDesc[0], hx:data(),
                 self.cxDesc[0], cx:data(),
                 self.wDesc[0], w:data(),
                 self.yDescs, y:data(),
                 self.hyDesc[0], hy:data(),
                 self.cyDesc[0], cy:data(),
                 self.workspace:data(), self.workspace:size(1) * 4) -- sizeof(float)
    end
end

function RNN:updateGradInput(input, gradOutput)
    assert(input:dim() == 3, 'Input should have three dimensions: (seqLength, miniBatch, inputSize)')
    assert(input:size(1) == self.seqLength, 'Sequence length has changed!')
    assert(input:size(2) == self.miniBatch, 'Minibatch size has changed!')
    assert(input:size(3) == self.inputSize, 'Input size has changed!')

    assert(gradOutput:isSameSizeAs(self.output), 'gradOutput has incorrect size!')
    assert(self.train, 'updateGradInput can only be called when training!')

    local x, dy = makeContiguous(self, input, gradOutput)
    local y = self.output
    local w = self.weight
    local hx = self.hiddenInput
    local dx = self.gradInput:resizeAs(input)

    if hx then
      assert(hx:dim() == 3, 'Hidden input must have 3 dimensions: (numLayers, miniBatch, hiddenSize)')
      assert(hx:size(1) == self.numLayers, 'Hidden input has incorrect number of layers!')
      assert(hx:size(2) == self.miniBatch, 'Hidden input has incorrect number of minibathes!')
      assert(hx:size(3) == self.hiddenSize, 'Hidden input has incorrect size!')
    end

    local cx = self.cellInput
    if cx then
      assert(cell:dim() == 3, 'Cell input must have 3 dimensions: (numLayers, miniBatch, hiddenSize)')
      assert(cell:size(1) == self.numLayers, 'Cell input has incorrect number of layers!')
      assert(cell:size(2) == self.miniBatch, 'Cell input has incorrect number of minibathes!')
      assert(cell:size(3) == self.hiddenSize, 'Cell input has incorrect size!')
    end

    local dhy = self.gradHiddenOutput
    if dhy then
      assert(hx:dim() == 3, 'Hidden output gradient must have 3 dimensions: (numLayers, miniBatch, hiddenSize)')
      assert(hx:size(1) == self.numLayers, 'Hidden output gradient has incorrect number of layers!')
      assert(hx:size(2) == self.miniBatch, 'Hidden output gradient has incorrect number of minibathes!')
      assert(hx:size(3) == self.hiddenSize, 'Hidden output gradient has incorrect size!')
    end

    local dcy = self.gradHiddenOutput
    if dcy then
      assert(cell:dim() == 3, 'Cell output gradient must have 3 dimensions: (numLayers, miniBatch, hiddenSize)')
      assert(cell:size(1) == self.numLayers, 'Cell output gradient has incorrect number of layers!')
      assert(cell:size(2) == self.miniBatch, 'Cell output gradient has incorrect number of minibathes!')
      assert(cell:size(3) == self.hiddenSize, 'Cell output gradient has incorrect size!')
    end

    local dhx = self.gradHiddenInput:resize(self.numLayers, self.miniBatch, self.hiddenSize)
    local dcx = self.gradCellInput:resize(self.numLayers, self.miniBatch, self.hiddenSize)

    errcheck('cudnnRNNBackwardData',
             cudnn.getHandle(),
             self.rnnDesc[0],
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
             self.workspace:data(), self.workspace:size(1) * 4, -- sizeof(float)
             self.reserve:data(), self.reserve:size(1) * 4) -- sizeof(float)
end

function RNN:accGradParameters(input, gradOutput, scale)
    assert(input:dim() == 3, 'Input should have three dimensions: (seqLength, miniBatch, inputSize)')
    assert(input:size(1) == self.seqLength, 'Sequence length has changed!')
    assert(input:size(2) == self.miniBatch, 'Minibatch size has changed!')
    assert(input:size(3) == self.inputSize, 'Input size has changed!')

    assert(gradOutput:isSameSizeAs(self.output), 'gradOutput has incorrect size!')
    assert(self.train, 'updateGradInput can only be called when training!')

    local x, dy = makeContiguous(self, input, gradOutput)
    local hx = self.hiddenInput
    local y = self.output
    local dw = self.gradWeight

    if hx then
      assert(hx:dim() == 3, 'Hidden input must have 3 dimensions: (numLayers, miniBatch, hiddenSize)')
      assert(hx:size(1) == self.numLayers, 'Hidden input has incorrect number of layers!')
      assert(hx:size(2) == self.miniBatch, 'Hidden input has incorrect number of minibathes!')
      assert(hx:size(3) == self.hiddenSize, 'Hidden input has incorrect size!')
    end

    if scale == 0 then
        return
    end

    -- cudnnRNNBackwardWeights doesn't accept a scale parameter so instead
    -- scale before and after.
    -- TODO: How much does this impact accuracy? Use a secondary buffer instead?
    if scale ~= 1 then
        local scaleTensor = torch.Tensor({1 / scale})
        errcheck('cudnnScaleTensor',
                 cudnn.getHandle(),
                 self.wDesc[0],
                 self.dw:data(),
                 scaleTensor:data())
    end
   
    errcheck('cudnnRNNBackwardWeights',
             cudnn.getHandle(),
             self.rnnDesc[0],
             self.xDescs, x:data(),
             self.hxDesc[0], hx and hx:data() or nil,
             self.yDescs, y:data(),
             self.workspace:data(), self.workspace:size(1) * 4, -- sizeof(float)
             self.wDesc[0], dw:data(),
             self.reserve:data(), self.reserve:size(1) * 4) -- sizeof(float)

    if scale ~= 1 then
        local scaleTensor = torch.Tensor({scale})
        errcheck('cudnnScaleTensor',
                 cudnn.getHandle(),
                 self.wDesc[0],
                 self.dw:data(),
                 scaleTensor:data())
    end
end

function RNN:clearDesc()
   self.dropoutDesc = nil
   self.rnnDesc = nil
   self.dropoutDesc = nil
   self.wDesc = nil
   self.xDescs = nil
   self.yDescs = nil
   self.hxDesc = nil
   self.hyDesc = nil
   self.cxDesc = nil
   self.cyDesc = nil
end

function RNN:write(f)
   self:clearDesc()
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function RNN:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_input', '_gradOutput',
                        'reserve', 'dropoutStates')
   return parent.clearState(self)
end
