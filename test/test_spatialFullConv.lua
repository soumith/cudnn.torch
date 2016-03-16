require 'cudnn'
require 'cunn'

local cudnntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local precision_jac = 1e-3
local nloop = 1
local times = {}
local mytester


local function testSpatialFullConv (imageWidth, imageHeight, nPlanesIn, nPlanesOut, kW, kH, dW, dH, padW, padH, adjW, adjH)

    print ("Running testSpatialFullConv (" ..
            "imageWidth = " .. imageWidth .. ", " ..
            "imageHeight = " .. imageHeight .. ", " ..
            "nPlanesIn = " .. nPlanesIn .. ", " ..
            "nPlanesOut = " .. nPlanesOut .. ", " ..
            "kW = " .. kW .. ", " ..
            "kH = " .. kH .. ", " ..
            "dW = " .. dW .. ", " ..
            "dH = " .. dH .. ", " ..
            "padW = " .. padW .. ", " ..
            "padH = " .. padH .. ", " ..
            "adjW = " .. adjW .. ", " ..
            "adjH = " .. adjH)

    local layerInput = torch.randn(1, nPlanesIn, imageHeight, imageWidth):cuda()

    local modelGT = nn.SpatialFullConvolution (nPlanesIn, nPlanesOut, kW, kH, dW, dH, padW, padH, adjW, adjH)
    local modelCUDNN = cudnn.SpatialFullConvolution (nPlanesIn, nPlanesOut, kW, kH, dW, dH, padW, padH, adjW, adjH)
    modelCUDNN.weight:copy (modelGT.weight)
    modelCUDNN.bias:copy (modelGT.bias)

    modelGT:cuda()
    modelCUDNN:cuda()

    local outputGT = modelGT:forward (layerInput)
    local outputCUDNN = modelCUDNN:forward (layerInput)

    local errorOutput = outputCUDNN:float() - outputGT:float()
    mytester:assertlt(errorOutput:abs():max(), precision_forward, 'error on state (forward) ')

    -- Now check the backwards diffs
    local crit = nn.MSECriterion()
    crit:cuda()
    local target = outputGT:clone()
    target:zero()
    target:cuda()

    local f = crit:forward (outputGT, target)
    local df_do = crit:backward (outputGT, target)

    local gradCUDNN = modelCUDNN:updateGradInput (layerInput, df_do)
    local gradGT = modelGT:updateGradInput (layerInput, df_do)
    local errorGradInput = gradCUDNN:float() - gradGT:float()
    mytester:assertlt(errorGradInput:abs():max(), precision_backward, 'error on grad input (backward) ')

    modelCUDNN:zeroGradParameters()
    modelCUDNN:accGradParameters (layerInput, df_do, 1.0)
    modelGT:zeroGradParameters()
    modelGT:accGradParameters (layerInput, df_do:cuda(), 1.0)

    local errorGradBias = (modelCUDNN.gradBias - modelGT.gradBias)
    mytester:assertlt(errorGradBias:abs():max(), precision_backward, 'error on grad bias (backward) ')

    local errorGradWeight = (modelCUDNN.gradWeight - modelGT.gradWeight)
    mytester:assertlt(errorGradWeight:abs():max(), precision_backward, 'error on grad weight (backward) ')
end

function cudnntest.SpatialConvolution_params()
    -- Test with a wide variety of different parameter values:
    testSpatialFullConv (5, 5, 1, 1, 3, 3, 2, 2, 0, 0, 0, 0)
    testSpatialFullConv (5, 5, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0)
    testSpatialFullConv (5, 7, 1, 1, 3, 1, 2, 2, 1, 1, 0, 0)
    testSpatialFullConv (7, 5, 1, 1, 3, 1, 1, 1, 1, 1, 0, 0)
    testSpatialFullConv (8, 5, 3, 1, 3, 3, 2, 2, 1, 1, 0, 0)
    testSpatialFullConv (5, 5, 1, 3, 3, 3, 2, 2, 1, 1, 0, 0)
    testSpatialFullConv (5, 5, 5, 3, 3, 3, 2, 2, 1, 1, 1, 1)
    testSpatialFullConv (9, 9, 3, 3, 3, 5, 2, 3, 0, 1, 1, 0)
end

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(cudnntest)

for i=1,cutorch.getDeviceCount() do
   print('Running test on device: ' .. i)
   cutorch.setDevice(i)
   mytester:run(tests)
end

os.execute('rm -f modelTemp.t7')
