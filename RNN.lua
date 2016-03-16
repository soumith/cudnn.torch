local RNN, parent = torch.class('cudnn.RNN', 'nn.Module')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function RNN:__init(hiddenSize, numLayers)
    parent.__init(self)

    self.datatype = 0 -- TODO CUDNN_FLOAT, should get the constant from ffi
    self.hiddenSize = hiddenSize
    self.inputSize = 0
    self.seqLength = 0
    self.numLayers = numLayers
    self.miniBatch = 0
    self.bidirectional = 0
    self.inputMode = 0 -- TODO CUDNN_LINEAR_INPUT, should get the constant from ffi
    self.mode = 0 -- TODO CUDNN_RNN_RELU, should get the constant from ffi
    self.dropout = 0
    self.seed = 0x01234567

    self.output = torch.CudaTensor()
    self.weight = torch.CudaTensor()
    self.gradWeight = torch.CudaTensor()
    self.hidden = torch.CudaTensor()
    self.cell = torch.CudaTensor()
    self.hy = torch.CudaTensor()
    self.cy = torch.CudaTensor()
    self.reserve = torch.CudaTensor(1)
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

local function createTensorDescriptors(count) return createDescriptors(count,
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
             0, -- TODO ffi CUDNN_TENSOR_NCHW
             3,
             dim:data())
end

function RNN:resetIODescriptors()
    self.xDescs = createTensorDescriptors(self.seqLength)
    self.dxDescs = createTensorDescriptors(self.seqLength)
    self.yDescs = createTensorDescriptors(self.seqLength)
    self.dyDescs = createTensorDescriptors(self.seqLength)

    for i = 0, self.seqLength - 1 do
        local dim = torch.IntTensor({self.inputSize, self.miniBatch, 1})
        local stride = torch.IntTensor({1, dim[1], dim[1] * dim[2]})

        errcheck('cudnnSetTensorNdDescriptor',
                 self.xDescs[i],
                 self.datatype,
                 3,
                 dim:data(),
                 stride:data())
        errcheck('cudnnSetTensorNdDescriptor',
                 self.dxDescs[i],
                 self.datatype,
                 3,
                 dim:data(),
                 stride:data())

        dim[1] = self.hiddenSize * (self.bidirectional > 0 and 2 or 1)
        stride[2] = dim[1]
        stride[3] = dim[1] * dim[2]

        errcheck('cudnnSetTensorNdDescriptor',
                 self.yDescs[i],
                 self.datatype,
                 3,
                 dim:data(),
                 stride:data())
        errcheck('cudnnSetTensorNdDescriptor',
                 self.dyDescs[i],
                 self.datatype,
                 3,
                 dim:data(),
                 stride:data())
    end
end

function RNN:resetHiddenDescriptors()
    self.hxDesc = cudnn.toDescriptor(self.hidden)
    self.hyDesc = cudnn.toDescriptor(self.hy)
    if self.dhx and self.dhy then
        self.dhxDesc = cudnn.toDescriptor(self.dhx)
        self.dhyDesc = cudnn.toDescriptor(self.dhy)
    end
end

function RNN:resetCellDescriptors()
    self.cxDesc = cudnn.toDescriptor(self.cell)
    self.cyDesc = cudnn.toDescriptor(self.cy)
    if self.dcx and self.dcy then
        self.dcxDesc = cudnn.toDescriptor(self.dcx)
        self.dcyDesc = cudnn.toDescriptor(self.dcy)
    end
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

function RNN:updateOutput(input)

    assert(input:dim() == 3)

    -- Decide which descriptors/tensors need to be updated.
    local resetRNN = not DropoutDesc or not RNNDesc
    local resetIO = not xDescs or not yDescs
    local resetHC = not self.hxDesc or not self.hyDesc or
                    not self.cxDesc or not self.cyDesc
    local resetWeight = not wDesc

    if input:size(1) ~= self.inputSize then
        self.inputSize = input:size(1)
        resetRNN = true
        resetIO = true
        resetWeight = true
    end

    if input:size(2) ~= self.miniBatch then
        self.miniBatch = input:size(1)
        resetRNN = true
        resetIO = true
        resetHC = true
        resetWeight = true
    end

    if input:size(3) ~= self.seqLength then
        self.seqLength = input:size(1)
        resetRNN = true
        resetIO = true
    end

    -- Update descriptors/tensors
    if resetRNN then
        self:resetDropoutDescriptor()
        self:resetRNNDescriptor()
    end

    local x = self:makeContiguous(input)
    local y = self.output
    if resetIO then
        self.output:resize(self.hiddenSize, self.miniBatch, self.seqLength)
        self:resetIODescriptors()
    end

    local hx = self.hidden
    local cx = self.cell
    local hy = self.hy
    local cy = self.cy
    if resetHC then
        self.hidden:resize(self.hiddenSize, self.miniBatch, self.numLayers)
        self.cell:resize(self.hiddenSize, self.miniBatch, self.numLayers)
        self.hy:resize(self.hiddenSize, self.miniBatch, self.numLayers)
        self.cy:resize(self.hiddenSize, self.miniBatch, self.numLayers)
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
             self.hxDesc[0], hx:data(),
             self.cxDesc[0], cx:data(),
             self.wDesc[0], w:data(),
             self.yDescs, y:data(),
             self.hyDesc[0], hy:data(),
             self.cyDesc[0], cy:data(),
             self.workspace:data(), self.workspace:size(1) * 4, -- sizeof(float)
             self.reserve:data(), self.reserve:size(1) * 4) -- sizeof(float)

    -- Hidden/cell output becomes the new hidden/cell input.
    self.hidden, self.hy = self.hy, self.hidden
    self.cell, self.cy = self.cy, self.cell
end

function RNN:updateGradInput(input, gradOutput)
    -- TODO prepare everything
    errcheck('cudnnRNNBackwardData',
             cudnn.getHandle(),
             self.rnnDesc[0],
             self.yDescs, y:data(),
             self.dyDescs, dy:data(),
             self.dhyDesc[0], dhy:data(),
             self.wDesc[0], w:data(),
             self.hxDesc[0], hx:data(),
             self.cxDesc[0], cx:data(),
             self.dxDescs, dx:data(),
             self.dhxDesc[0], dhx:data(),
             self.dcxDesc[0], dcx:data(),
             self.workspace:data(), self.workspace:size(1),
             self.reserve:data(), self.reserve:size(1))
    
end

function RNN:accGradParameters()
    -- TODO prepare everything
    errcheck('cudnnRNNBackwardWeights',
             cudnn.getHandle(),
             self.rnnDesc[0],
             self.xDescs, x:data(),
             self.hxDesc[0], hx:data(),
             self.yDescs, y:data(),
             self.workspace:data(), self.workspace:size(1),
             self.dwDesc[0], dw:data(),
             self.reserve:data(), self.reserve:size(1))
end

