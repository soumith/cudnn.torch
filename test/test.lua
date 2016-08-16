require 'cudnn'
require 'cunn'

local cudnntest = torch.TestSuite()
local nloop = 1
local times = {}
local mytester
local jac = nn.Jacobian


local testparams_half = {
   test_type = 'torch.CudaHalfTensor',
   precision_forward = 2e-1,
   precision_backward = 6,
   precision_jac = 1e-3,
   precision_io = 1e-1,
}

local testparams_float = {
   test_type = 'torch.CudaTensor',
   precision_forward = 1e-4,
   precision_backward = 2e-2,
   precision_jac = 1e-3,
   precision_io = 1e-5,
}

-- TODO: find out why the errors are so huge
local testparams_double = {
   test_type = 'torch.CudaDoubleTensor',
   precision_forward = 1e-4,
   precision_backward = 2e-2,
   precision_jac = 1e-3,
   precision_io = 1e-5,
}

local testparams = nil

local function cast(input)
   return input:type(testparams.test_type)
end

-- workarounds
function torch.CudaHalfTensor:__sub(b)
   return self:cuda() - b:cuda()
end

function torch.CudaHalfTensor:abs()
   return self:cuda():abs():cudaHalf()
end

function torch.CudaDoubleTensor:abs()
   return self:cuda():abs():cudaDouble()
end

function torch.CudaHalfTensor:mean()
   return self:cuda():mean()
end

function torch.CudaDoubleTensor:__sub(b)
   return self:cuda() - b:cuda()
end

function torch.CudaDoubleTensor:mean()
   return self:cuda():mean()
end

-- whapper for math.random() to accomodate certain limitations for 'half'
local function random2(min,max, half_max)
   if half_max and testparams.test_type == 'torch.CudaHalfTensor' then
      return math.random(min,half_max)
   else
      return math.random(min,max)
   end
end

local function testLayer(nnlayer, cudnnlayer, input, gradOutput, scale,
                         parametric, batchMode, description)
   description = description or ''
   -- serialize and deserialize
   torch.save('modelTemp.t7', cudnnlayer)
   cudnnlayer = torch.load('modelTemp.t7')

   if not batchMode then -- convert given mini-batch to single sample
      input = input[1]:clone()
      gradOutput = gradOutput[1]:clone()
   end
   local gt = {} -- groundtruth
   gt.output = nnlayer:forward(input)
   nnlayer:zeroGradParameters()
   gt.gradInput = nnlayer:backward(input, gradOutput, scale)
   if parametric then
      gt.gradWeight = nnlayer.gradWeight
      gt.gradBias = nnlayer.gradBias
   end

   local res = {} -- result
   res.output = cudnnlayer:forward(cast(input))
   cudnnlayer:zeroGradParameters()
   res.gradInput = cudnnlayer:backward(cast(input), cast(gradOutput), scale)
   if parametric then
      res.gradWeight = cudnnlayer.gradWeight
      res.gradBias = cudnnlayer.gradBias
   end

   for name, _ in pairs(gt) do
      local error = gt[name]:float() - res[name]:float()
      error = error:abs():max()
      local precision
      if name == 'output' then
         precision = testparams.precision_forward
      else
         precision = testparams.precision_backward
      end
      mytester:assertlt(error, precision, 'error on ' .. name
                           .. ', batchMode = ' .. tostring(batchMode)
                           .. ', type = ' .. torch.type(res[name])
                           .. ', ' .. description)
   end

   -- IO
   local ferr,berr = jac.testIO(cudnnlayer, cast(input))
   mytester:assertlt(ferr, testparams.precision_io,
                     torch.typename(cudnnlayer) .. ' - i/o forward err '
                        .. ', batchMode = ' .. tostring(batchMode)
                        .. ', type = ' .. torch.type(res[name])
                        .. ', ' .. description)
   mytester:assertlt(berr, testparams.precision_io,
                     torch.typename(cudnnlayer) .. ' - i/o backward err '
                        .. ', batchMode = ' .. tostring(batchMode)
                        .. ', type = ' .. torch.type(res[name])
                        .. ', ' .. description)
end

function cudnntest.SpatialConvolution()
   local bs = random2(1,32)
   local from = random2(1,32)
   local to = random2(1,64)
   local ki = random2(1,9,4)
   local kj = random2(1,9,4)
   local si = random2(1,ki, 1)
   local sj = random2(1,kj, 1)
   local outi = random2(1,64)
   local outj = random2(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,to,outj,outi):cuda()
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   local gconv = cast(cudnn.SpatialConvolution(from,to,ki,kj,si,sj))

   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)

   testLayer(sconv, gconv, input, gradOutput, scale, true, true) -- batch
   testLayer(sconv, gconv, input, gradOutput, scale, true, false) -- non-batch
   local originalTypename = torch.typename(gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv),
                     originalTypename, 'conversion type check')
   testLayer(sconv, gconv, input, gradOutput, scale, true, true)
   testLayer(sconv, gconv, input, gradOutput, scale, true, false)
end

function cudnntest.SpatialFullConvolution()

   local bs = random2(1,32)
   local from = random2(1,32)
   local to = random2(1,64)
   local ki = random2(1,9)
   local kj = random2(1,9)
   local si = random2(1,ki,1)
   local sj = random2(1,kj,1)
   local ini = random2(1,64)
   local inj = random2(1,64)
   local outi = (ini-1)*si+ki
   local outj = (inj-1)*sj+kj
   local scale = math.random()

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,to,outj,outi):cuda()
   local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj):cuda()
   local gconv = cast(cudnn.SpatialFullConvolution(from,to,ki,kj,si,sj):cuda():fastest())
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)

   testLayer(sconv, gconv, input, gradOutput, scale, true, true) -- batch
   testLayer(sconv, gconv, input, gradOutput, scale, true, false) -- non-batch
   local originalTypename = torch.typename(gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv),
                     originalTypename, 'conversion type check')
   testLayer(sconv, gconv, input, gradOutput, scale, true, true)
   testLayer(sconv, gconv, input, gradOutput, scale, true, false)
end

function cudnntest.TemporalConvolution()
   if testparams.test_type == 'torch.CudaHalfTensor' then
      -- was not able to foind any parameters that would work for half
      return
   end
   local bs = random2(2,32)
   local inputFrameSize = random2(1,64)
   local outputFrameSize = random2(1,64)
   local ki = random2(2,6)
   local si = random2(2,ki,1)
   local outi = random2(2,9,4)
   local ini = (outi - 1) * si + ki
   local scale = math.random()

   local input = torch.randn(bs,ini,inputFrameSize):cuda()
   local gradOutput = torch.randn(bs,outi,outputFrameSize):cuda()
   local sconv = nn.TemporalConvolution(inputFrameSize,outputFrameSize, ki, si):cuda()
   local gconv = cast(cudnn.TemporalConvolution(inputFrameSize,outputFrameSize, ki, si):cuda():fastest())
   gconv.weight:copy(sconv.weight:view(gconv.weight:size()))
   gconv.bias:copy(sconv.bias)

   testLayer(sconv, gconv, input, gradOutput, scale, true, true) -- batch
   testLayer(sconv, gconv, input, gradOutput, scale, true, false) -- non-batch
   -- temporal convolution does not support cudnn.convert, so no tests for that
end

function cudnntest.TemporalConvolution_padding_batch()
   if testparams.test_type == 'torch.CudaHalfTensor' then
      -- was not able to foind any parameters that would work for half
      return
   end
   local bs = random2(2,32)
   local inputFrameSize = random2(2,64)
   local outputFrameSize = random2(2,64)
   local ki = random2(2,9)
   local pad_h = math.floor(ki/2)
   local si = random2(1,ki,1)
   local outi = random2(2,9,4)
   local ini = (outi-1)*si+ki
   local scale = math.random()

   local inputpadded = torch.randn(bs,ini,inputFrameSize):cuda()
   for i=1,pad_h do
      inputpadded:narrow(2,i,1):fill(0)
      inputpadded:narrow(2,ini-i+1,1):fill(0)
   end
   local input = torch.Tensor(bs,ini - 2 * pad_h, inputFrameSize):cuda()
   input:copy(inputpadded:narrow(2, pad_h+1, ini - 2 * pad_h))
   local gradOutput = torch.randn(bs,outi,outputFrameSize):cuda()
   local sconv = nn.TemporalConvolution(inputFrameSize,outputFrameSize, ki, si):cuda()
   local groundForward = sconv:forward(inputpadded)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(inputpadded, gradOutput, scale)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cast(cudnn.TemporalConvolution(inputFrameSize,outputFrameSize, ki, si,pad_h):cuda():fastest())
   gconv.weight:copy(sconv.weight:view(gconv.weight:size()))
   gconv.bias:copy(sconv.bias)
   gconv:forward(cast(input))

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local cudaForward = gconv:forward(cast(input))
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(cast(input), cast(gradOutput), scale)
   cutorch.synchronize()
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias

   local ferror = cudaForward:float() - groundForward:float()
   groundgrad = groundgrad:narrow(2, pad_h + 1, ini - 2 * pad_h)
   local error = rescuda:float() - groundgrad:float()
   local werror = weightcuda:float() - groundweight:float()
   local berror = biascuda:float() - groundbias:float()
   mytester:assertlt(ferror:abs():max(), testparams.precision_forward, 'error on forward  ')
   mytester:assertlt(error:abs():max(), testparams.precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), testparams.precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), testparams.precision_backward, 'error on bias (backward) ')
end

function cudnntest.TemporalConvolution_reduceBatchSize()
   local inputFrameSize = random2(1,64)
   local outputFrameSize = random2(1,64)
   local ki = random2(1,9,6)
   local si = random2(1,ki,1)
   local outi = random2(2,9)
   local ini = (outi-1)*si+ki
   local batchSize = 128
   local smallerBatchSize = batchSize/2

   local input = cast(torch.randn(batchSize,ini,inputFrameSize))
   local conv = cast(cudnn.TemporalConvolution(inputFrameSize,outputFrameSize,ki,si):cuda())
   local o1 = conv:updateOutput(input)
   mytester:asserteq(o1:size(1), batchSize, 'batch size didn\'t match')

   input = cast(torch.randn(smallerBatchSize,ini,inputFrameSize))
   local o2 = conv:updateOutput(input)
   mytester:asserteq(o2:size(1), smallerBatchSize, 'batch size didn\'t match')
   -- do this again to check it doesn't crash
   local o2 = conv:updateOutput(input)
   mytester:asserteq(o2:size(1), smallerBatchSize, 'batch size didn\'t match')
end

function cudnntest.VolumetricConvolution()
   if testparams.test_type == 'torch.CudaHalfTensor' then
      -- was not able to foind any parameters that would work for half
      return
   end
   local bs = random2(1,32)
   local from = random2(1,bs)
   local to = random2(from,bs)
   local ki = random2(3,5,3)
   local kj = random2(3,5,3)
   local kk = random2(3,5,3)
   local si = random2(1,ki-1,1)
   local sj = random2(1,kj-1,1)
   local sk = random2(1,kk-1,1)
   local outi = random2(2,17)
   local outj = random2(2,17)
   local outk = random2(2,5)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk
   local scale = math.random()

   local input = torch.randn(bs,from,ink,inj,ini):cuda()
   local gradOutput = torch.randn(bs,to,outk,outj,outi):cuda()
   local sconv = nn.VolumetricConvolution(from,to,kk,ki,kj,sk,si,sj):cuda()
   local gconv = cast(cudnn.VolumetricConvolution(from,to,kk,ki,kj,sk,si,sj))
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)

   testLayer(sconv, gconv, input, gradOutput, scale, true, true) -- batch
   testLayer(sconv, gconv, input, gradOutput, scale, true, false) -- non-batch
   local originalTypename = torch.typename(gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv),
                     originalTypename, 'conversion type check')
   testLayer(sconv, gconv, input, gradOutput, scale, true, true)
   testLayer(sconv, gconv, input, gradOutput, scale, true, false)
end

function cudnntest.VolumetricMaxPooling()
   local bs = random2(1,4)
   local from = random2(1,4)
   local ki = random2(2,4)
   local kj = random2(2,4)
   local kk = random2(2,4)
   local si = ki
   local sj = kj
   local sk = kk
   local outi = random2(1,64)
   local outj = random2(1,64)
   local outk = random2(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   local input = torch.randn(bs,from,ink,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outk,outj,outi):cuda()
   local sconv = nn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj):cuda()
   local gconv = cast(cudnn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj))

   testLayer(sconv, gconv, input, gradOutput, scale, false, true) -- batch
   testLayer(sconv, gconv, input, gradOutput, scale, false, false) -- non-batch
   local originalTypename = torch.typename(gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv),
                     originalTypename, 'conversion type check')
   testLayer(sconv, gconv, input, gradOutput, scale, false, true)
   testLayer(sconv, gconv, input, gradOutput, scale, false, false)
end

function cudnntest.SpatialMaxPooling()
   local bs = random2(1,32)
   local from = random2(1,32)
   local ki = random2(2,4)
   local kj = random2(2,4)
   local si = random2(1,4)
   local sj = random2(1,4)
   local outi = random2(16,64)
   local outj = random2(16,64)
   local padi = random2(0,ki/2-1)
   local padj = random2(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = random2(0,1) == 1

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):cuda()
   if ceil_mode then sconv:ceil() end
   local gconv = cast(cudnn.SpatialMaxPooling(ki,kj,si,sj,padi,padj))
   if ceil_mode then gconv:ceil() end

   testLayer(sconv, gconv, input, gradOutput, scale, false, true) -- batch
   testLayer(sconv, gconv, input, gradOutput, scale, false, false) -- non-batch
   local originalTypename = torch.typename(gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv),
                     originalTypename, 'conversion type check')
   testLayer(sconv, gconv, input, gradOutput, scale, false, true)
   testLayer(sconv, gconv, input, gradOutput, scale, false, false)
end

function cudnntest.SpatialAveragePooling()
   local bs = random2(1,32)
   local from = random2(1,32)
   local ki = random2(2,4)
   local kj = random2(2,4)
   local si = random2(2,4)
   local sj = random2(2,4)
   local outi = random2(1,64)
   local outj = random2(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()
   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   local gconv = cast(cudnn.SpatialAveragePooling(ki,kj,si,sj))

   testLayer(sconv, gconv, input, gradOutput, scale, false, true) -- batch
   testLayer(sconv, gconv, input, gradOutput, scale, false, false) -- non-batch
   local originalTypename = torch.typename(gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv),
                     originalTypename, 'conversion type check')
   testLayer(sconv, gconv, input, gradOutput, scale, false, true)
   testLayer(sconv, gconv, input, gradOutput, scale, false, false)

   mytester:assert(cudnn.C.CUDNN_POOLING_AVERAGE ~= nil, 'back-compat broken')
end

local function nonlin(nonlin, inplace)
   local bs = random2(1,32)
   local from = random2(1,32)
   local outi = random2(1,64)
   local outj = random2(1,64)
   local ini = outi
   local inj = outj

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()
   local sconv = nn[nonlin](inplace):cuda()
   local gconv = cast(cudnn[nonlin](inplace))

   local description = 'inplace = ' .. tostring(inplace)
   testLayer(sconv, gconv, input, gradOutput, scale, false, true, description)
   testLayer(sconv, gconv, input, gradOutput, scale, false, false, description)
   local originalTypename = torch.typename(gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv),
                     originalTypename, 'conversion type check')
   testLayer(sconv, gconv, input, gradOutput, scale, false, true, description)
   testLayer(sconv, gconv, input, gradOutput, scale, false, false, description)
end

function cudnntest.ReLU()
   nonlin('ReLU', true) -- inplace
   nonlin('ReLU', false) -- out of place
end
function cudnntest.Tanh()
   nonlin('Tanh', false) -- out of place
end
function cudnntest.Sigmoid()
   nonlin('Sigmoid', false) -- out of place
end

function cudnntest.ClippedReLU_single()
    local input = torch.randn(1, 32):cuda()
    local ceiling = 0.1
    local module = cudnn.ClippedReLU(ceiling):cuda()
    local output = module:forward(input)
    local expectedOutput = input:clone()
    expectedOutput[expectedOutput:ge(ceiling)] = ceiling
    expectedOutput[expectedOutput:le(0)] = 0
    mytester:assertTensorEq(output, expectedOutput)
end

function cudnntest.SpatialCrossMapLRN_batch()
   local bs = random2(4,10)
   local inputSize = random2(6,9)
   local size = random2(1,3)*2+1
   local nbfeatures = random2(3,8)
   local alpha = random2(1,100)/100
   local beta  = random2(0,100)/100
   local k = random2(1,3)

   local input = torch.rand(bs, nbfeatures, inputSize, inputSize):cuda()
   local gradOutput = torch.rand(input:size()):cuda()
   local sconv = nn.SpatialCrossMapLRN(size, alpha, beta, k):cuda()
   local gconv = cast(cudnn.SpatialCrossMapLRN(size, alpha, beta, k))

   testLayer(sconv, gconv, input, gradOutput, scale, true, true) -- batch
   testLayer(sconv, gconv, input, gradOutput, scale, true, false) -- non-batch
   local originalTypename = torch.typename(gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv),
                     originalTypename, 'conversion type check')
   testLayer(sconv, gconv, input, gradOutput, scale, true, true)
   testLayer(sconv, gconv, input, gradOutput, scale, true, false)
end

function cudnntest.SoftMax_single()
   local bs = random2(1, 32)
   local sz = random2(1,64)
   local input = torch.randn(bs, sz):cuda()
   local gradOutput = torch.randn(bs, sz):cuda()

   local sconv = nn.SoftMax():cuda()
   local gconv = cast(cudnn.SoftMax())

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   testLayer(sconv, gconv, input, gradOutput, scale, false, true) -- batch
   testLayer(sconv, gconv, input, gradOutput, scale, false, false) -- non-batch
   local originalTypename = torch.typename(gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv),
                     originalTypename, 'conversion type check')
   testLayer(sconv, gconv, input, gradOutput, scale, false, true)
   testLayer(sconv, gconv, input, gradOutput, scale, false, false)
end

function cudnntest.LogSoftMax()
   local bs = random2(1, 32)
   local sz = random2(1,64)
   local input = torch.randn(bs, sz):cuda()
   local gradOutput = torch.randn(bs, sz):cuda()

   local sconv = nn.LogSoftMax():cuda()
   local gconv = cast(cudnn.LogSoftMax())

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   testLayer(sconv, gconv, input, gradOutput, scale, false, true) -- batch
   testLayer(sconv, gconv, input, gradOutput, scale, false, false) -- non-batch
   local originalTypename = torch.typename(gconv)
   local gconv = cast(cudnn.convert(sconv, cudnn))
   mytester:asserteq(torch.typename(gconv),
                     originalTypename, 'conversion type check')
   testLayer(sconv, gconv, input, gradOutput, scale, false, true)
   testLayer(sconv, gconv, input, gradOutput, scale, false, false)
end

function cudnntest.SpatialLogSoftMax()
    -- batch
    local numLabels = random2(5,10)
    local h = random2(5,10)
    local w = random2(5,10)
    local bsz = random2(3, 7)
    local input = torch.zeros(bsz, numLabels, h, w):normal():cuda()
    local target = torch.zeros(bsz, numLabels, h, w):normal():cuda()

    local cri = cast(cudnn.SpatialLogSoftMax())
    local gcri = nn.LogSoftMax():cuda()

    local op = cri:forward(cast(input), cast(target))
    local gi = cri:backward(cast(input), cast(target))

    local gop = op:clone():zero()
    local ggi = gi:clone():zero()

    for i=1,h do
        for j=1,w do
            local i1 = input[{{}, {}, {i}, {j}}]:contiguous():squeeze()
            local t1 = target[{{}, {}, {i}, {j}}]:contiguous():squeeze()
            local gop1 = gcri:forward(i1, t1)
            local ggi1 = gcri:backward(i1, t1)
            gop[{{}, {}, {i}, {j}}]:copy(gop1)
            ggi[{{}, {}, {i}, {j}}]:copy(ggi1)
        end
    end
    local err = (gi - ggi):abs():max()
    mytester:assertlt(err, testparams.precision_backward,
                      'error in difference between central difference and :backward')
    local err = (op - gop):abs():max()
    mytester:assertlt(err, testparams.precision_backward,
                      'error in difference between central difference and :backward')
end

local function testBatchNormalization(moduleName, inputSize)
   local input = torch.randn(table.unpack(inputSize)):cuda()
   local gradOutput = torch.randn(table.unpack(inputSize)):cuda()
   local cbn = cast(cudnn[moduleName](inputSize[2], 1e-3))
   local gbn = nn[moduleName](inputSize[2], 1e-3):cuda()
   cbn.weight:copy(gbn.weight)
   cbn.bias:copy(gbn.bias)

   local function testFWDBWD(cbn, gbn)
      cbn:training()
      gbn:training()
      mytester:asserteq(cbn.running_mean:mean(), 0, 'error on BN running_mean init')
      mytester:asserteq(cbn.running_var:mean(), 1, 'error on BN running_var init')
      local rescuda = cbn:forward(cast(input))
      local groundtruth = gbn:forward(input)
      local resgrad = cbn:backward(cast(input), cast(gradOutput))
      local groundgrad = gbn:backward(input, gradOutput)

      local error = rescuda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (forward) ')
      error = resgrad:float() - groundgrad:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_backward, 'error in batch normalization (backward) ')
      error = cbn.running_mean:float() - gbn.running_mean:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (running_mean) ')
      error = cbn.running_var:float() - gbn.running_var:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (running_var) ')
   end

   local function testFWD(cbn, gbn)
      cbn:evaluate()
      gbn:evaluate()
      local rescuda = cbn:forward(cast(input))
      local groundtruth = gbn:forward(input)

      local error = rescuda:float() - groundtruth:float()
      mytester:assertlt(error:abs():max(),
         testparams.precision_forward, 'error in batch normalization (forward) ')
   end

   testFWDBWD(cbn, gbn)
   testFWD(cbn, gbn)
   if testparams.test_type == 'torch.CudaTensor' then
      local cudnn2nn = cast(cudnn.convert(cbn:clone(), nn))
      mytester:asserteq(torch.type(cudnn2nn), 'nn.'..moduleName, 'cudnn to nn')
      testFWD(cudnn2nn, gbn)
      local nn2cudnn = cast(cudnn.convert(gbn:clone(), cudnn))
      mytester:asserteq(torch.type(nn2cudnn), 'cudnn.'..moduleName, 'cudnn to nn')
      testFWD(nn2cudnn, gbn)
   end
end

function cudnntest.BatchNormalization()
   local size = {
      random2(2, 32),
      random2(16, 256),
   }
   testBatchNormalization('BatchNormalization', size)
end

function cudnntest.SpatialBatchNormalization()
   local size = {
      random2(1, 32),
      random2(1, 32),
      random2(5, 10),
      random2(5, 10),
   }
   testBatchNormalization('SpatialBatchNormalization', size)
end

function cudnntest.VolumetricBatchNormalization()
   local size = {
      random2(1, 32),
      random2(1, 32),
      random2(2, 6),
      random2(2, 6),
      random2(2, 6),
   }
   testBatchNormalization('VolumetricBatchNormalization', size)
end

function cudnntest.SpatialCrossEntropyCriterion()
    if testparams.test_type ~= 'torch.CudaTensor' then return end
    -- batch
    local numLabels = random2(5,10)
    local h = random2(5,10)
    local w = random2(5,10)
    local bsz = random2(3, 7)
    local input = torch.zeros(bsz, numLabels, h, w):normal():cuda()
    local target = torch.Tensor(bsz, h, w):random(1, numLabels):cuda()

    local cri = cast(cudnn.SpatialCrossEntropyCriterion())
    local gcri = nn.CrossEntropyCriterion():cuda()

    local op = cri:forward(cast(input), cast(target))
    local gi = cri:backward(cast(input), cast(target))

    local ggi = gi:clone():zero()

    for i=1,h do
        for j=1,w do
            local i1 = input[{{}, {}, {i}, {j}}]:contiguous():squeeze()
            local t1 = target[{{}, {i}, {j}}]:contiguous():squeeze()
            local gop1 = gcri:forward(i1, t1)
            local ggi1 = gcri:backward(i1, t1)
            ggi[{{}, {}, {i}, {j}}]:copy(ggi1)
        end
    end

    -- nn.CrossEntropy in contrast to cudnn.SpatialCrossEntropyCriterion cannot
    -- average over the last spatial dimensions because it is run in a loop
    ggi:div(h * w)

    local err = (gi - ggi):abs():max()
    mytester:assertlt(err, testparams.precision_backward,
                      'error in difference between central difference and :backward')
end

function cudnntest.functional_bias2D()
   local bs = random2(1,32)
   local from = random2(1,32)
   local to = random2(1,64)
   local ki = random2(1,15)
   local kj = random2(1,15)
   local si = random2(1,ki)
   local sj = random2(1,kj)
   local outi = random2(1,64)
   local outj = random2(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local scale = torch.uniform()
   local input = torch.zeros(bs,from,inj,ini):cuda()
   local mod = cudnn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   mod.weight:zero()
   local groundtruth = mod:forward(input)
   local result = groundtruth:clone():zero()
   cudnn.functional.bias2D_updateOutput(cudnn.getHandle(), mod.bias, result)
   local error = result:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(),
                     testparams.precision_forward, 'error on forward ')

   mod:zeroGradParameters()
   local gradOutput = groundtruth:clone():normal()
   mod:backward(input, gradOutput, scale)
   local groundtruth = mod.gradBias
   local result = groundtruth:clone():zero()
   cudnn.functional.bias2D_accGradParameters(cudnn.getHandle(), gradOutput, result, scale)
   error = result:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(),
                     testparams.precision_backward, 'error on accGradParameters ')
end

function cudnntest.functional_convolution2d()
    local a=cudnn.SpatialConvolution(3,16,5,5):cuda()
    a.bias:zero();
    local input = torch.randn(10,3,10,10):cuda()
    a:zeroGradParameters()
    a:forward(input);
    local output = a.output:clone():normal()
    local gradOutput = a.output:clone():normal()
    local gradInput = a:backward(input, gradOutput):clone():normal()
    local gradWeight = a.gradWeight:clone():zero()
    cudnn.functional.Convolution2D_updateOutput(cudnn.getHandle(), input,
                                                a.weight, output, a.dH,
                                                a.dW, a.padH, a.padW)
    mytester:assertlt((output - a.output):abs():max(),
                     testparams.precision_forward, 'error on forward ')

    cudnn.functional.Convolution2D_updateGradInput(cudnn.getHandle(), input,
                                                   a.weight, output, gradOutput,
                                                   gradInput,
                                                   a.dH, a.dW, a.padH, a.padW)
    mytester:assertlt((gradInput - a.gradInput):abs():max(),
                     testparams.precision_forward, 'error on updateGradInput ')

    cudnn.functional.Convolution2D_accGradParameters(cudnn.getHandle(), input,
                                                   gradWeight, gradOutput,
                                                   a.dH, a.dW, a.padH, a.padW)
    mytester:assertlt((gradWeight - a.gradWeight):abs():max(),
                     testparams.precision_forward, 'error on accGradParameters ')
end

function cudnntest.functional_maxpooling2d()
    local a=cudnn.SpatialMaxPooling(2,2,2,2):cuda()
    local input = torch.randn(10,3,10,10):cuda()
    a:forward(input);
    local output = a.output:clone():normal()
    local gradOutput = a.output:clone():normal()
    local gradInput = a:backward(input, gradOutput):clone():normal()
    cudnn.functional.MaxPooling2D_updateOutput(cudnn.getHandle(), input,
                                               output, a.kH, a.kW,
                                               a.dH, a.dW, a.padH, a.padW)
    mytester:assertlt((output - a.output):abs():max(),
                     testparams.precision_forward, 'error on forward ')

    cudnn.functional.MaxPooling2D_updateGradInput(cudnn.getHandle(), input,
                                                   output, gradOutput, gradInput,
                                                   a.kH, a.kW, a.dH, a.dW,
                                                   a.padH, a.padW)
    mytester:assertlt((gradInput - a.gradInput):abs():max(),
                     testparams.precision_forward, 'error on updateGradInput ')
end

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(cudnntest)

cudnn.verbose=false

-- Developers, do not commit uncommented regions until bindings fixed
-- local runHalf = true

for i = 1, cutorch.getDeviceCount() do

   for _, benchmark in ipairs({true, false}) do
      cudnn.benchmark = benchmark
      local prop = cutorch.getDeviceProperties(i)

      print('Running test on device: #' .. i .. ' : ' .. prop.name
               .. ' with benchmark = ' .. tostring(cudnn.benchmark))

      cutorch.setDevice(i)

      if runHalf and cudnn.benchmark then
         print'Testing torch.CudaHalfTensor'
         testparams = testparams_half
         mytester:run()
      else
         print(
            [[Half Tensor tests are disabled due to missing functionality.
They will be enabled once fully fixed and functional.
See https://github.com/soumith/cudnn.torch/issues/225 for progress
]])
      end

      print'Testing torch.CudaTensor'
      testparams = testparams_float
      mytester:run()

      print'Testing torch.CudaDoubleTensor'
      testparams = testparams_double
      mytester:run()


   end
end

os.execute('rm -f modelTemp.t7')
