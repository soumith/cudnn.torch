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

    self.gradInput = torch.CudaTensor()
    self.output = torch.CudaTensor()
    self.weight = torch.CudaTensor()
    self.gradParameters = torch.CudaTensor()
    self.hx = torch.CudaTensor()
    self.cx = torch.CudaTensor()
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
    self.yDescs = createTensorDescriptors(self.seqLength)

    for i = 0, self.seqLength - 1 do
        local dim = torch.IntTensor({self.inputSize, self.miniBatch, 1})
        local stride = torch.IntTensor({1, dim[1], dim[1] * dim[2]})

        errcheck('cudnnSetTensorNdDescriptor',
                 self.xDescs[i],
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
    end
end

function RNN:resetHiddenDescriptors()
    self.hxDesc = cudnn.toDescriptor(self.hx)
    self.hyDesc = cudnn.toDescriptor(self.hy)
end

function RNN:resetCellDescriptors()
    self.cxDesc = cudnn.toDescriptor(self.cx)
    self.cyDesc = cudnn.toDescriptor(self.cy)
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

    -- Hidden/cell output becomes the new hidden/cell input.
    local hx = self.hy
    local cx = self.cy
    local hy = self.hx
    local cy = self.cx
    if resetHC then
        self.hx:resize(self.hiddenSize, self.miniBatch, self.numLayers)
        self.cx:resize(self.hiddenSize, self.miniBatch, self.numLayers)
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
end

function RNN:updateGradInput(input, gradOutput)

    assert(input:dim() == 3)
    assert(input:size(1) == self.inputSize)
    assert(input:size(2) == self.miniBatch)
    assert(input:size(3) == self.seqLength)

    assert(gradOutput:dim() == self.output:dim())
    for i = 1, gradOutput:dim() do
        assert(gradOutput:size(i) == self.output:size(i))
    end

    local y = self.output
    local dy = gradOutput
    local w = self.weight
    local hx = self.hx
    local cx = self.cx
    local dx = self.gradInput

    if dx:dim() ~= 3 or
       dx:size(1) ~= input:size(1) or
       dx:size(2) ~= input:size(2) or
       dx:size(3) ~= input:size(3) then
        dx:resizeAs(input)
    end

    errcheck('cudnnRNNBackwardData',
             cudnn.getHandle(),
             self.rnnDesc[0],
             self.yDescs, y:data(),
             self.yDescs, dy:data(),
             self.hyDesc[0], nil, -- TODO should dhy be ignored?
             self.cyDesc[0], nil, -- TODO should dhy be ignored?
             self.wDesc[0], w:data(),
             self.hxDesc[0], hx:data(),
             self.cxDesc[0], cx:data(),
             self.xDescs, dx:data(),
             self.hxDesc[0], nil, -- TODO should dhx be ignored?
             self.cxDesc[0], nil, -- TODO should dcx be ignored?
             self.workspace:data(), self.workspace:size(1) * 4, -- sizeof(float)
             self.reserve:data(), self.reserve:size(1) * 4) -- sizeof(float)
end

function RNN:accGradParameters(input, gradOutput, scale)

    assert(input:dim() == 3)
    assert(input:size(1) == self.inputSize)
    assert(input:size(2) == self.miniBatch)
    assert(input:size(3) == self.seqLength)

    assert(gradOutput:dim() == self.output:dim())
    for i = 1, gradOutput:dim() do
        assert(gradOutput:size(i) == self.output:size(i))
    end

    local x = input
    local hx = self.hx
    local y = self.output
    local dw = self.gradParameters

    if dw:dim() ~= 3 or
       dw:size(1) ~= self.weight:size(1) or
       dw:size(2) ~= self.weight:size(2) or
       dw:size(3) ~= self.weight:size(3) then
        dw:resizeAs(self.weight)
    end

    if scale == 0 then
        return
    end

    -- cudnnRNNBackwardWeights doesn't accept a scale parameter so instead
    -- scale before and after.
    -- TODO: How much does this impact accuracy?
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
             self.hxDesc[0], hx:data(),
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

