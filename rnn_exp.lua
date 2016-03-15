import 'cudnn'
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local datatype = 0      -- TODO CUDNN_FLOAT, should get the constant from ffi
local hiddenSize = 1    -- TODO This is a layer parameter, correct?
local inputSize = 1     -- TODO Is this a layer parameter or determined by input?
local seqLength = 1     -- TODO Is this a layer parameter or determined by input?
local numLayers = 1     -- TODO
local miniBatch = 1     -- TODO
local bidirectional = 0 -- TODO CUDNN_UNIDIRECTIONAL, should get the constant from ffi
local inputMode = 0     -- TODO CUDNN_LINEAR_INPUT, should get the constant from ffi
local mode = 0          -- TODO CUDNN_RNN_RELU, should get the constant from ffi
local dropout = 0       -- TODO 
local seed = 0x01234567 -- TODO 

-- Dropout Descriptor

local dropoutStatesSize = torch.LongTensor(1)
errcheck('cudnnDropoutGetStatesSize',
         cudnn.getHandle(),
         dropoutStatesSize:data())
local dropoutStates = torch.CudaTensor(dropoutStatesSize[1])

local dropoutDesc = ffi.new('cudnnDropoutDescriptor_t[?]', 1)
errcheck('cudnnCreateDropoutDescriptor', dropoutDesc)
-- TODO GC was being called early. Ignore cleanup for now.
-- ffi.gc(dropoutDesc, function(d) errcheck('cudnnDestroyDropoutDescriptor', d[0]) end)
errcheck('cudnnSetDropoutDescriptor',
         dropoutDesc[0],
         cudnn.getHandle(),
         dropout,
                 -- TODO Using dropoutStates causes an invalid memory access error.
         nil, 0, -- dropoutStates:data(), dropoutStatesSize[1],
         seed)

-- RNN Descriptor
local rnnDesc = ffi.new('cudnnRNNDescriptor_t[?]', 1)
errcheck('cudnnCreateRNNDescriptor', rnnDesc)
-- ffi.gc(rnnDesc, function(d) errcheck('cudnnDestroyRNNDescriptor', d[0]) end)
errcheck('cudnnSetRNNDescriptor',
         rnnDesc[0],
         hiddenSize,
         seqLength,
         numLayers,
         dropoutDesc[0],
         inputMode,
         bidirectional,
         mode,
         datatype)

-- Input
local inputDescs = ffi.new('cudnnTensorDescriptor_t[?]', seqLength)
for i = 0, seqLength - 1 do
    errcheck('cudnnCreateTensorDescriptor', inputDescs + i)
end
-- ffi.gc(inputDescs, function()
--     for i = 0, seqLength - 1 do
--         errcheck('cudnnDestroyTensorDescriptor', inputDescs[i])
--     end
-- end)

local dims = torch.IntTensor({inputSize, miniBatch, seqLength})
local stride = torch.IntTensor({1, dims[1], 1})

for i = 0, seqLength - 1 do
    errcheck('cudnnSetTensorNdDescriptor',
             inputDescs[i],
             datatype,
             3,
             dims:data(),
             stride:data())
end

local input = torch.CudaTensor(dims[1], dims[2], dims[3])

-- Ouptut
local outputDescs = ffi.new('cudnnTensorDescriptor_t[?]', seqLength)
for i = 0, seqLength - 1 do
    errcheck('cudnnCreateTensorDescriptor', outputDescs + i)
end
-- ffi.gc(outputDescs, function()
--     for i = 0, seqLength - 1 do
--         errcheck('cudnnDestroyTensorDescriptor', outputDescs[i])
--     end
-- end)

local dims = torch.IntTensor({hiddenSize, miniBatch, seqLength})
local stride = torch.IntTensor({1, dims[1], 1})

for i = 0, seqLength - 1 do
    errcheck('cudnnSetTensorNdDescriptor',
             inputDescs[i],
             datatype,
             3,
             dims:data(),
             stride:data())
end

local output = torch.CudaTensor(dims[1], dims[2], dims[3])

-- Hidden
local hiddenInputDesc = ffi.new('cudnnTensorDescriptor_t[?]', 1)
local hiddenOutputDesc = ffi.new('cudnnTensorDescriptor_t[?]', 1)
errcheck('cudnnCreateTensorDescriptor', hiddenInputDesc)
errcheck('cudnnCreateTensorDescriptor', hiddenOutputDesc)
-- ffi.gc(hiddenInputDesc, function(d) errcheck('cudnnDestroyTensorDescriptor', d[0]) end)
-- ffi.gc(hiddenOutputDesc, function(d) errcheck('cudnnDestroyTensorDescriptor', d[0]) end)

local dims = torch.IntTensor({hiddenSize, miniBatch, numLayers})
local stride = torch.IntTensor({1, dims[1], 1})

errcheck('cudnnSetTensorNdDescriptor',
         hiddenInputDesc[0],
         datatype,
         3,
         dims:data(),
         stride:data())
errcheck('cudnnSetTensorNdDescriptor',
         hiddenOutputDesc[0],
         datatype,
         3,
         dims:data(),
         stride:data())

local hiddenInput = torch.CudaTensor(dims[1], dims[2], dims[3])
local hiddenOutput = torch.CudaTensor(dims[1], dims[2], dims[3])

-- Cell
local cellInputDesc = ffi.new('cudnnTensorDescriptor_t[?]', 1)
local cellOutputDesc = ffi.new('cudnnTensorDescriptor_t[?]', 1)
errcheck('cudnnCreateTensorDescriptor', cellInputDesc)
errcheck('cudnnCreateTensorDescriptor', cellOutputDesc)
-- ffi.gc(cellInputDesc, function(d) errcheck('cudnnDestroyTensorDescriptor', d[0]) end)
-- ffi.gc(cellOutputDesc, function(d) errcheck('cudnnDestroyTensorDescriptor', d[0]) end)

local dims = torch.IntTensor({hiddenSize, miniBatch, numLayers})
local stride = torch.IntTensor({1, dims[1], 1})

errcheck('cudnnSetTensorNdDescriptor',
         cellInputDesc[0],
         datatype,
         3,
         dims:data(),
         stride:data())
errcheck('cudnnSetTensorNdDescriptor',
         cellOutputDesc[0],
         datatype,
         3,
         dims:data(),
         stride:data())

local cellInput = torch.CudaTensor(dims[1], dims[2], dims[3])
local cellOutput = torch.CudaTensor(dims[1], dims[2], dims[3])

-- Weight
local weightDesc = ffi.new('cudnnFilterDescriptor_t[?]', 1)
errcheck('cudnnCreateFilterDescriptor', weightDesc)
-- ffi.gc(weightDesc, function(d) errcheck('cudnnDestroyFilterDescriptor', d[0]) end)

local weightSize = torch.LongTensor(1)
errcheck('cudnnGetRNNParamsSize',
         cudnn.getHandle(),
         rnnDesc[0],
         inputDescs,
         weightSize:data())
local dims = torch.IntTensor({weightSize[1] / 4, 1, 1}) -- sizeof(float)

errcheck('cudnnSetFilterNdDescriptor',
         weightDesc[0],
         datatype,
         0, -- TODO ffi CUDNN_TENSOR_NCHW
         3,
         dims:data())
local weight = torch.CudaTensor(dims[1], dims[2], dims[3])

-- Workspace
local workspace = cudnn.getSharedWorkspace()
local workspaceSize = torch.LongTensor(1)
errcheck('cudnnGetRNNWorkspaceSize',
         cudnn.getHandle(),
         rnnDesc[0],
         inputDescs,
         workspaceSize:data())
workspace:resize(workspaceSize[1] / 4) -- sizeof(float)

-- Print Descriptor data
print("hiddenSize = " .. hiddenSize)
print("inputSize = " .. inputSize)
print("seqLength = " .. seqLength)
print("numLayers = " .. numLayers)
print("miniBatch = " .. miniBatch)
print("bidirectional = " .. bidirectional)
print("inputMode = " .. inputMode)
print("mode = " .. mode)
print("dropout = " .. dropout)

local datatype = torch.IntTensor(1)
local nbDims = torch.IntTensor(1)
local dims = torch.IntTensor(3)
local stride = torch.IntTensor(3)

errcheck('cudnnGetTensorNdDescriptor',
         inputDescs[0],
         3,
         datatype:data(),
         nbDims:data(),
         dims:data(),
         stride:data())
print("Input " ..
      "dim=(" .. dims[1] .. ", " .. dims[2] .. ", " .. dims[3] .. ") " ..
      "stride=(" .. stride[1] .. ", " .. stride[2] .. ", " .. stride[3] .. ")")

errcheck('cudnnGetTensorNdDescriptor',
         outputDescs[0],
         3,
         datatype:data(),
         nbDims:data(),
         dims:data(),
         stride:data())
print("Output " .. 
      "dim=(" .. dims[1] .. ", " .. dims[2] .. ", " .. dims[3] .. ") " ..
      "stride=(" .. stride[1] .. ", " .. stride[2] .. ", " .. stride[3] .. ")")

errcheck('cudnnGetTensorNdDescriptor',
         hiddenInputDesc[0],
         3,
         datatype:data(),
         nbDims:data(),
         dims:data(),
         stride:data())
print("Hidden Input " .. 
      "dim=(" .. dims[1] .. ", " .. dims[2] .. ", " .. dims[3] .. ") " ..
      "stride=(" .. stride[1] .. ", " .. stride[2] .. ", " .. stride[3] .. ")")

errcheck('cudnnGetTensorNdDescriptor',
         hiddenOutputDesc[0],
         3,
         datatype:data(),
         nbDims:data(),
         dims:data(),
         stride:data())
print("Hidden Output " .. 
      "dim=(" .. dims[1] .. ", " .. dims[2] .. ", " .. dims[3] .. ") " ..
      "stride=(" .. stride[1] .. ", " .. stride[2] .. ", " .. stride[3] .. ")")

errcheck('cudnnGetTensorNdDescriptor',
         cellInputDesc[0],
         3,
         datatype:data(),
         nbDims:data(),
         dims:data(),
         stride:data())
print("Cell Input " .. 
      "dim=(" .. dims[1] .. ", " .. dims[2] .. ", " .. dims[3] .. ") " ..
      "stride=(" .. stride[1] .. ", " .. stride[2] .. ", " .. stride[3] .. ")")

errcheck('cudnnGetTensorNdDescriptor',
         cellOutputDesc[0],
         3,
         datatype:data(),
         nbDims:data(),
         dims:data(),
         stride:data())
print("Cell Output " .. 
      "dim=(" .. dims[1] .. ", " .. dims[2] .. ", " .. dims[3] .. ") " ..
      "stride=(" .. stride[1] .. ", " .. stride[2] .. ", " .. stride[3] .. ")")

local format = ffi.new('cudnnTensorFormat_t[?]', 1)
errcheck('cudnnGetFilterNdDescriptor',
         weightDesc[0],
         3,
         datatype:data(),
         format,
         nbDims:data(),
         dims:data())

print("Weight " .. 
      "dim=(" .. dims[1] .. ", " .. dims[2] .. ", " .. dims[3] .. ") ")

-- ForwardInference
errcheck('cudnnRNNForwardInference',
         cudnn.getHandle(),
         rnnDesc[0],
         inputDescs, input:data(),
         hiddenInputDesc[0], nil, -- hiddenInput:data(),
         cellInputDesc[0], nil, -- cellInput:data(),
         weightDesc[0], weight:data(),
         outputDescs, output:data(),
         hiddenOutputDesc[0], nil, -- hiddenOutput:data(),
         cellOutputDesc[0], nil, -- cellOutput:data(),
         workspace:data(), workspace:size(1) * 4) -- sizeof(float)

