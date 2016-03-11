local RNN, parent = torch.class('cudnn.RNN', 'nn.Module')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function RNN:__init() -- TODO parameters
    parent.__init(self)

    self.datatype = 0      -- TODO CUDNN_FLOAT, should get the constant from ffi
    self.hiddenSize = 1    -- TODO This is a layer parameter, correct?
    self.inputSize = 1     -- TODO Is this a layer parameter or determined by input?
    self.seqLength = 1     -- TODO Is this a layer parameter or determined by input?
    self.numLayers = 1     -- TODO
    self.miniBatch = 1     -- TODO
    self.bidirectional = 0 -- TODO
    self.inputMode = 0     -- TODO CUDNN_LINEAR_INPUT, should get the constant from ffi
    self.mode = 0          -- TODO CUDNN_RNN_RELU, should get the constant from ffi
    self.dropout = 0       -- TODO 
    self.seed = 0x01234567 -- TODO 

    -- TODO should these be CudaTensors ?
    self.weight = torch.CudaTensor() -- TODO size
    self.gradWeight = torch.CudaTensor() -- TODO size
    self.hx = torch.CudaTensor() -- TODO size (should this be internal/external?)
    self.cx = torch.CudaTensor() -- TODO size (should this be internal/external?)
    self.hy = torch.CudaTensor() -- TODO size (should this be internal/external?)
    self.cy = torch.CudaTensor() -- TODO size (should this be internal/external?)

    self.h0 = torch.CudaTensor() -- TODO input hidden copy from output
    self.c0 = torch.CudaTensor() -- TODO input cell copy from cell

    self.reserve = torch.CudaTensor() -- TODO
end

local function createDescriptors(count, descs_type, create_func, destroy_func)
    local ds = ffi.new(descs_type, count)
    for i = 0, count - 1 do
        errcheck(create_func, ds + i)
    end
    local function destroyDescriptors(ds)
        for i = 0, count - 1 do
            errcheck(destroy_func, ds + i)
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

function RNN:resetDropoutDescriptors()
    if not self.dropoutDesc then
        self.dropoutDesc = createDropoutDescriptors(1)
    end
    errcheck('cudnnSetDropoutDescriptor',
             self.dropoutDesc[0],
             cudnn.getHandle(),
             self.dropout,
             nil, 0, -- TODO self.dropoutStates, self.dropoutStatesSize,
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
    local weightSize = ffi.new("size_t[1]")
    errcheck('cudnnGetRNNParamsSize',
             cudnn.getHandle(),
             self.rnnDesc[0],
             self.xDescs,
             weightSize)
    dim = torch.IntTensor(3)
    dim[1] = tonumber(weightSize[0])
    dim[2] = 1
    dim[3] = 1
    errcheck('cudnnSetFilterNdDescriptor',
             self.wDesc[0],
             self.datatype,
             0, -- TODO ffi CUDNN_TENSOR_NCHW
             3,
             dim:data())
end

function RNN:resetIODescriptors()
    if not self.xDescs or not self.dxDescs or -- TODO seqLength might change?
       not self.yDescs or not self.dyDescs then
        self.xDescs = createTensorDescriptors(self.seqLength)
        self.dxDescs = createTensorDescriptors(self.seqLength)
        self.yDescs = createTensorDescriptors(self.seqLength)
        self.dyDescs = createTensorDescriptors(self.seqLength)
    end
    for i = 0, self.seqLength - 1 do
        -- TODO miniBatch shrink?
        local currentMiniBatch = self.miniBatch

        -- TODO is the following always correct?
        local dim = torch.IntTensor(3)
        dim[1] = self.inputSize
        dim[2] = currentMiniBatch
        dim[3] = 1

        -- TODO Is stride used by cudnn RNN functions?
        local stride = torch.IntTensor(3)
        stride[1] = 1
        stride[2] = dim[1]
        stride[3] = 1

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

function RNN:resetHCDescriptors()
    if not self.hxDesc or not self.dhxDesc or
       not self.hyDesc or not self.dhyDesc or
       not self.cxDesc or not self.dcxDesc or
       not self.cyDesc or not self.dcyDesc then
        self.hxDesc = createTensorDescriptors(1)
        self.dhxDesc = createTensorDescriptors(1)
        self.hyDesc = createTensorDescriptors(1)
        self.dhyDesc = createTensorDescriptors(1)
        self.cxDesc = createTensorDescriptors(1)
        self.dcxDesc = createTensorDescriptors(1)
        self.cyDesc = createTensorDescriptors(1)
        self.dcyDesc = createTensorDescriptors(1)
    end

    -- TODO is the following always correct?
    local dim = torch.IntTensor(3)
    dim[1] = self.hiddenSize
    dim[2] = self.miniBatch
    dim[3] = self.numLayers

    -- TODO Is stride used by cudnn RNN functions?
    local stride = torch.IntTensor(3)
    stride[1] = 1
    stride[2] = self.hiddenSize
    stride[3] = 1

    local function fill(desc)
        errcheck('cudnnSetTensorNdDescriptor',
                 desc,
                 self.datatype,
                 3,
                 dim:data(),
                 stride:data())
    end
    fill(self.hxDesc[0])
    fill(self.dhxDesc[0])
    fill(self.hyDesc[0])
    fill(self.dhyDesc[0])
    fill(self.cxDesc[0])
    fill(self.dcxDesc[0])
    fill(self.cyDesc[0])
    fill(self.dcyDesc[0])
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
    --TODO this?
    --self.miniBatch = input:size(1) -- ?
    --self.seqLength = input:size(2) -- ?
    --self.inputSize = input:size(3) -- ?

    --TODO or this?
    assert(input:size(1) == self.miniBatch)
    assert(input:size(2) == self.seqLength)
    assert(input:size(3) == self.inputSize)

    -- TODO Which need to be done every iteration and which can be done only once?
    self:resetDropoutDescriptors()
    self:resetRNNDescriptor()
    self:resetIODescriptors()
    self:resetHCDescriptors()
    self:resetWeightDescriptors()

    local x = self:makeContiguous(input)
    self.hx:resize(self.numLayers, self.miniBatch, self.hiddenSize) -- TODO correct?
    local hx = self.hx
    self.cx:resize(self.numLayers, self.miniBatch, self.hiddenSize) -- TODO correct?
    local cx = self.cx

    self.output:resize(self.miniBatch, self.seqLength, self.hiddenSize) -- TODO correct?
    local y = self.output
    self.hy:resize(self.numLayers, self.miniBatch, self.hiddenSize) -- TODO correct?
    local hy = self.hy -- TODO should this equal hx (in place?)
    self.cy:resize(self.numLayers, self.miniBatch, self.hiddenSize) -- TODO correct?
    local cy = self.cy -- TODO should this equal cx (in place?)

    local weightSize = ffi.new("size_t[1]")
    errcheck('cudnnGetRNNParamsSize',
             cudnn.getHandle(),
             self.rnnDesc[0],
             self.xDescs,
             weightSize)
    self.weight:resize(tonumber(weightSize[0]))
    local w = self.weight

    self.workspace = cudnn.getSharedWorkspace()
    local workspaceSize = ffi.new("size_t[1]")
    errcheck('cudnnGetRNNWorkspaceSize',
             cudnn.getHandle(),
             self.rnnDesc[0],
             self.xDescs,
             workspaceSize)
    self.workspace:resize(tonumber(workspaceSize[0]))

    local reserveSize = ffi.new("size_t[1]")
    errcheck('cudnnGetRNNTrainingReserveSize',
             cudnn.getHandle(),
             self.rnnDesc[0],
             self.xDescs,
             reserveSize)
    self.reserve:resize(tonumber(reserveSize[0]))

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
             self.workspace:data(), self.workspace:size(1),
             self.reserve:data(), self.reserve:size(1))
end

function RNN:updateGradInput()
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

