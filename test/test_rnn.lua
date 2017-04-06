--[[
--  Tests the implementation of RNN binding using the cudnn v5 library. Cross-check the checksums with cudnn reference
--  sample checksums.
-- ]]

require 'cudnn'
require 'cunn'
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local cudnntest = torch.TestSuite()
local mytester

local tolerance = 1000

function cudnntest.testRNNRELU()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local rnn = cudnn.RNNReLU(hiddenSize, hiddenSize, numberOfLayers)
    rnn.mode = 'CUDNN_RNN_RELU'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)

    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.315793E+06, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.315212E+05, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 6.676003E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 6.425067E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.453750E+09, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testRNNBatchFirst()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local batchFirst = true
    local rnn = cudnn.RNNReLU(hiddenSize, hiddenSize, numberOfLayers, batchFirst)
    rnn.mode = 'CUDNN_RNN_RELU'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst)

    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.315793E+06, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.315212E+05, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 6.676003E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 6.425067E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.453750E+09, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testRNNTANH()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local rnn = cudnn.RNNTanh(hiddenSize, hiddenSize, numberOfLayers)
    rnn.mode = 'CUDNN_RNN_TANH'
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)

    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 6.319591E+05, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 6.319605E+04, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 4.501830E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 4.489546E+00, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 5.012598E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testRNNLSTM()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 8
    local rnn = cudnn.LSTM(hiddenSize, hiddenSize, numberOfLayers)
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)

    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 5.749536E+05, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumc, 4.365091E+05, tolerance, 'checkSum with reference for localSumc failed')
    mytester:assertalmosteq(checkSums.localSumh, 5.774818E+04, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 3.842206E+02, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdc, 9.323785E+03, tolerance, 'checkSum with reference for localSumdc failed')
    mytester:assertalmosteq(checkSums.localSumdh, 1.182566E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 4.313461E+08, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testRNNGRU()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 6
    local rnn = cudnn.GRU(hiddenSize, hiddenSize, numberOfLayers)
    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn)
    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 6.358978E+05, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 6.281680E+04, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 6.296622E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 2.289960E+05, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 5.397419E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testBiDirectionalRELURNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local nbDirections = 2
    local batchFirst = false
    local rnn = cudnn.RNN(hiddenSize, hiddenSize, numberOfLayers)
    rnn.bidirectional = 'CUDNN_BIDIRECTIONAL'
    rnn.mode = 'CUDNN_RNN_RELU'
    rnn.numDirections = 2

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.388634E+01, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.288997E+01, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 1.288729E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 1.279004E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 7.061081E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testBiDirectionalTANHRNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 2
    local nbDirections = 2
    local batchFirst = false
    local rnn = cudnn.RNN(hiddenSize, hiddenSize, numberOfLayers)
    rnn.bidirectional = 'CUDNN_BIDIRECTIONAL'
    rnn.mode = 'CUDNN_RNN_TANH'
    rnn.numDirections = 2

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 1.388634E+01, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.288997E+01, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 1.288729E+01, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 1.279004E+01, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 7.061081E+07, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testBiDirectionalLSTMRNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 8
    local nbDirections = 2
    local batchFirst = false
    local rnn = cudnn.BLSTM(hiddenSize, hiddenSize, numberOfLayers)

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 3.134097E+04, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumc, 3.845626E+00, tolerance, 'checkSum with reference for localSumc failed')
    mytester:assertalmosteq(checkSums.localSumh, 1.922855E+00, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 4.794993E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdc, 2.870925E+04, tolerance, 'checkSum with reference for localSumdc failed')
    mytester:assertalmosteq(checkSums.localSumdh, 2.468645E+00, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.121568E+08, tolerance, 'checkSum with reference for localSumdw failed')
end

function cudnntest.testBiDirectionalGRURNN()
    local miniBatch = 64
    local seqLength = 20
    local hiddenSize = 512
    local numberOfLayers = 2
    local numberOfLinearLayers = 6
    local nbDirections = 2
    local batchFirst = false
    local rnn = cudnn.RNN(hiddenSize, hiddenSize, numberOfLayers)
    rnn.bidirectional = 'CUDNN_BIDIRECTIONAL'
    rnn.mode = 'CUDNN_GRU'
    rnn.numDirections = 2

    local checkSums = getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    -- Checksums to check against are retrieved from cudnn RNN sample.
    mytester:assertalmosteq(checkSums.localSumi, 6.555183E+04, tolerance, 'checkSum with reference for localsumi failed')
    mytester:assertalmosteq(checkSums.localSumh, 5.830924E+00, tolerance, 'checkSum with reference for localSumh failed')
    mytester:assertalmosteq(checkSums.localSumdi, 4.271801E+00, tolerance, 'checkSum with reference for localSumdi failed')
    mytester:assertalmosteq(checkSums.localSumdh, 6.555744E+04, tolerance, 'checkSum with reference for localSumdh failed')
    mytester:assertalmosteq(checkSums.localSumdw, 1.701796E+08, tolerance, 'checkSum with reference for localSumdw failed')
end

--[[
-- Method gets Checksums of RNN to compare with ref Checksums in cudnn RNN C sample.
-- ]]
function getRNNCheckSums(miniBatch, seqLength, hiddenSize, numberOfLayers, numberOfLinearLayers, rnn, batchFirst, nbDirections)
    local biDirectionalScale = nbDirections or 1
    -- Reset the rnn and weight descriptor (since we are manually setting values for matrix/bias.
    rnn:reset()
    rnn:resetWeightDescriptor()
    local input
    if (batchFirst) then
        input = torch.CudaTensor(miniBatch, seqLength, hiddenSize):fill(1)
    else
        input = torch.CudaTensor(seqLength, miniBatch, hiddenSize):fill(1) -- Input initialised to 1s.
    end
    local weights = rnn:weights()
    local biases = rnn:biases()
    -- Matrices are initialised to 1 / matrixSize, biases to 1 unless bi-directional.
    for layer = 1, numberOfLayers do
        for layerId = 1, numberOfLinearLayers do
            if (biDirectionalScale == 2) then
                rnn.weight:fill(1 / rnn.weight:size(1))
            else
                local weightTensor = weights[layer][layerId]
                weightTensor:fill(1.0 / weightTensor:size(1))

                local biasTensor = biases[layer][layerId]
                biasTensor:fill(1)
            end
        end
    end
    -- Set hx/cx/dhy/dcy data to 1s.
    rnn.hiddenInput = torch.CudaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    rnn.cellInput = torch.CudaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    rnn.gradHiddenOutput = torch.CudaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    rnn.gradCellOutput = torch.CudaTensor(numberOfLayers * biDirectionalScale, miniBatch, hiddenSize):fill(1)
    local testOutputi = rnn:forward(input)
    -- gradInput set to 1s.
    local gradInput
    if(batchFirst) then
        gradInput = torch.CudaTensor(miniBatch, seqLength, hiddenSize * biDirectionalScale):fill(1)
    else
        gradInput = torch.CudaTensor(seqLength, miniBatch, hiddenSize * biDirectionalScale):fill(1)
    end
    rnn:backward(input, gradInput)

    -- Sum up all values for each.
    local localSumi = torch.sum(testOutputi)
    local localSumh = torch.sum(rnn.hiddenOutput)
    local localSumc = torch.sum(rnn.cellOutput)

    local localSumdi = torch.sum(rnn.gradInput)
    local localSumdh = torch.sum(rnn.gradHiddenInput)
    local localSumdc = torch.sum(rnn.gradCellInput)

    local localSumdw = torch.sum(rnn.gradWeight)

    local checkSums = {
        localSumi = localSumi,
        localSumh = localSumh,
        localSumc = localSumc,
        localSumdi = localSumdi,
        localSumdh = localSumdh,
        localSumdc = localSumdc,
        localSumdw = localSumdw
    }
    return checkSums
end

function cudnntest.testPackPadSequences()
    -- T is 4, B = 5, vector size = 3
    local input = torch.CudaIntTensor({
        {{101, 102, 103},
         {201, 202, 203},
         {301, 302, 303},
         {401, 402, 403},
         {501, 502, 503}},
        {{104, 105, 106},
         {204, 205, 206},
         {304, 305, 306},
         {  0,   0,   0},
         {  0,   0,   0}},
        {{107, 108, 109},
         {207, 208, 209},
         {  0,   0,   0},
         {  0,   0,   0},
         {  0,   0,   0}},
        {{110, 111, 112},
         {  0,   0,   0},
         {  0,   0,   0},
         {  0,   0,   0},
         {  0,   0,   0}},
    })
    local lengths = {4, 3, 2, 1, 1}

    local expectedPacked = torch.CudaIntTensor({
        {101, 102, 103}, {201, 202, 203}, {301, 302, 303}, {401, 402, 403}, {501, 502, 503},
        {104, 105, 106}, {204, 205, 206}, {304, 305, 306},
        {107, 108, 109}, {207, 208, 209},
        {110, 111, 112}
    })
    local expectedBSPT = {5, 3, 2, 1}

    local result = cudnn.RNN:packPaddedSequence(input, lengths)
    local actualPacked, actualBSPT = unpack(result)
    mytester:assertTensorEq(expectedPacked, actualPacked)
    mytester:assertTableEq(expectedBSPT, actualBSPT)

    local actualUnpacked, actualLengths = cudnn.RNN:padPackedSequence(result)
    mytester:assertTensorEq(input, actualUnpacked)
    mytester:assertTableEq(lengths, actualLengths)

    -- test again with batchFirst
    input = input:transpose(1, 2)

    local result = cudnn.RNN:packPaddedSequence(input, lengths, true)
    local actualPacked, actualBSPT = unpack(result)
    mytester:assertTensorEq(expectedPacked, actualPacked)
    mytester:assertTableEq(expectedBSPT, actualBSPT)

    local actualUnpacked, actualLengths = cudnn.RNN:padPackedSequence(result, true)
    mytester:assertTensorEq(input, actualUnpacked)
    mytester:assertTableEq(lengths, actualLengths)
end

-- clone the parameters of src into dest, assumes both RNNs were created with
-- the same options (e.g. same input size, hidden size, layers, etc.)
local function deepcopyRNN(dest, src)
   dest.weight = src.weight:clone() -- encompasses W_hh, W_xh etc.
   dest.gradWeight = src.gradWeight:clone()
end

function cudnntest.testVariableLengthSequences()
   local input = torch.CudaTensor({
      {{1, 2, 2, 1},
       {2, 1, 2, 2},
       {1, 1, 1, 2},
       {2, 2, 2, 1}},
      {{4, 1, 3, 1},
       {3, 1, 2, 1},
       {1, 1, 2, 1},
       {0, 0, 0, 0}},
      {{1, 1, 2, 1},
       {2, 1, 2, 2},
       {1, 2, 2, 1},
       {0, 0, 0, 0}},
      {{1, 2, 1, 1},
       {0, 0, 0, 0},
       {0, 0, 0, 0},
       {0, 0, 0, 0}}
   })

   -- same as above
   local indivInputs = {
      torch.CudaTensor({
         {{1, 2, 2, 1}},
         {{4, 1, 3, 1}},
         {{1, 1, 2, 1}},
         {{1, 2, 1, 1}},
      }),
      torch.CudaTensor({
         {{2, 1, 2, 2}},
         {{3, 1, 2, 1}},
         {{2, 1, 2, 2}},
      }),
      torch.CudaTensor({
         {{1, 1, 1, 2}},
         {{1, 1, 2, 1}},
         {{1, 2, 2, 1}},
      }),
      torch.CudaTensor({
         {{2, 2, 2, 1}},
      }),
   }

   local lengths = {4, 3, 3, 1}
   local maxLength = 4

   -- Generate gradOutput based on input sizes
   local gradOutput = torch.CudaTensor(11, 1, 10):uniform()
   local indivGradOutputs = {
      torch.cat({gradOutput:narrow(1, 1, 1), gradOutput:narrow(1, 5, 1), gradOutput:narrow(1, 8, 1), gradOutput:narrow(1, 11, 1)}, 1):clone(),
      torch.cat({gradOutput:narrow(1, 2, 1), gradOutput:narrow(1, 6, 1), gradOutput:narrow(1, 9, 1)}, 1):clone(),
      torch.cat({gradOutput:narrow(1, 3, 1), gradOutput:narrow(1, 7, 1), gradOutput:narrow(1, 10, 1)}, 1):clone(),
      gradOutput:narrow(1, 4, 1):clone()
   }
   gradOutput = gradOutput:squeeze()

   local inputSize = 4
   local hiddenSize = 10
   local numLayers = 1
   local batchFirst = false
   local dropout = false
   local rememberStates = false

   local lstm = cudnn.LSTM(
      inputSize,
      hiddenSize,
      numLayers,
      batchFirst,
      dropout,
      rememberStates)

   local lstm2 = cudnn.LSTM(
      inputSize,
      hiddenSize,
      numLayers,
      batchFirst,
      dropout,
      rememberStates)

   deepcopyRNN(lstm2, lstm)

   -- Step 1: Pass Sequences as batch and individually, verify weights, outputs
   -- are the same in both instances

   -- batched
   local packed = cudnn.RNN:packPaddedSequence(input, lengths)
   local packedOutput = lstm:updateOutput(packed)
   local packedHiddenOutput = lstm.hiddenOutput:clone()
   -- could use padPackedSequence here, but for testing simplicity, we'll just
   -- operate on the returned results

   local separate = {}
   local hids = {}
   local indivGradInputs = {}

   for i, length in ipairs(lengths) do
      local inp = indivInputs[i]
      local output = lstm2:updateOutput(inp):clone()
      table.insert(separate, output)
      local hid = lstm2.hiddenOutput:clone()
      table.insert(hids, hid)

      -- need to do backwards pass here too
      local gradOutput = indivGradOutputs[i]
      local gradInp = lstm2:updateGradInput(inp, gradOutput):clone()
      table.insert(indivGradInputs, gradInp)
   end
   separate = torch.cat(separate, 1):squeeze()
   hids = torch.cat(hids, 1):squeeze()

   mytester:asserteq(packedOutput:size(1), separate:size(1))
   mytester:asserteq(packedOutput:size(2), separate:size(2))

   -- packedOutput has format where all 4 from first batch, then all 3 from
   -- second batch, etc. while separate has all 4 from first sequence,
   -- all 3 from next sequence, etc. I manually map the matches here
   local corresponding = {
      {1, 1},
      {2, 5},
      {3, 8},
      {4, 11},
      {5, 2},
      {6, 6},
      {7, 9},
      {8, 3},
      {9, 7},
      {10, 10},
      {11, 4}
   }
   for _, pair in ipairs(corresponding) do
      local sep, batched = unpack(pair)
      local diff = torch.csub(separate[sep], packedOutput[batched]):abs():sum()
      mytester:assert(diff < 1e-7)
   end

   local hdiff = torch.csub(packedHiddenOutput, hids):abs():sum()
   mytester:assert(hdiff < 1e7)

   -- Step 2: update grad input as batch and individually

   local packedGradInput = lstm:updateGradInput(packed, gradOutput)
   local igiTestable = torch.cat(indivGradInputs, 1):squeeze(2)

   for _, pair in ipairs(corresponding) do
      sep, batched = unpack(pair)
      local diff = torch.csub(igiTestable[sep], packedGradInput[batched]):abs():sum()
      mytester:assert(diff < 1e-7)
   end

   -- Step 3: Basically verify that accGradParameters works for batch
   lstm:accGradParameters(packed, gradOutput)
end

mytester = torch.Tester()
mytester:add(cudnntest)
mytester:run()
