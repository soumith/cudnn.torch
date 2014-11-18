require 'cudnn'
require 'cunn'

local cudnntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local precision_jac = 1e-3
local nloop = 1
local times = {}


function cudnntest.SpatialConvolution_forward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(bs,from,inj,ini):cuda()
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cuda()
   local groundtruth = sconv:forward(input)
   cutorch.synchronize()
   local gconv = cudnn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   local rescuda = gconv:forward(input)
   cutorch.synchronize()
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end


function cudnntest.SpatialConvolution_backward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,to,outj,outi):cuda()
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cuda()
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cudnn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias

   local error = rescuda:float() - groundgrad:float()
   local werror = weightcuda:float() - groundweight:float()
   local berror = biascuda:float() - groundbias:float()

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cudnntest.SpatialConvolution_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(from,inj,ini):cuda()
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cuda()
   local groundtruth = sconv:forward(input)
   cutorch.synchronize()
   local gconv = cudnn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   local rescuda = gconv:forward(input)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end


function cudnntest.SpatialConvolution_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(to,outj,outi):cuda()
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cuda()
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cudnn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias

   local error = rescuda:float() - groundgrad:float()
   local werror = weightcuda:float() - groundweight:float()
   local berror = biascuda:float() - groundbias:float()

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end


function cudnntest.SpatialMaxPooling_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj):cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.SpatialMaxPooling(ki,kj,si,sj):cuda()
   local rescuda = gconv:forward(input)
   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')
   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cudnntest.SpatialMaxPooling_single()
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(from,outj,outi):cuda()

   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj):cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.SpatialMaxPooling(ki,kj,si,sj):cuda()
   local rescuda = gconv:forward(input)
   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')
   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 3, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cudnntest.ReLU_single()
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(from,outj,outi):cuda()

   local sconv = nn.ReLU():cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.ReLU():cuda()
   local rescuda = gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 3, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cudnntest.ReLU_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn.ReLU():cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.ReLU():cuda()
   local rescuda = gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cudnntest.Tanh_single()
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(from,outj,outi):cuda()

   local sconv = nn.Tanh():cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.Tanh():cuda()
   local rescuda = gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 3, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cudnntest.Tanh_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn.Tanh():cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.Tanh():cuda()
   local rescuda = gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cudnntest.Sigmoid_single()
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(from,outj,outi):cuda()

   local sconv = nn.Sigmoid():cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.Sigmoid():cuda()
   local rescuda = gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 3, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cudnntest.Sigmoid_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn.Sigmoid():cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.Sigmoid():cuda()
   local rescuda = gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cudnntest.SoftMax_single()
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(from,outj,outi):cuda()

   local sconv = nn.SoftMax():cuda()
   local groundtruth = sconv:forward(input:view(-1))
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.SoftMax():cuda()
   local rescuda = gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 3, 'error in dimension')

   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(),
                     precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(),
                     precision_backward, 'error on state (backward) ')
end

function cudnntest.SoftMax_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn.SoftMax():cuda()
   local groundtruth = sconv:forward(input:view(bs,-1))
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.SoftMax():cuda()
   local rescuda = gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 4, 'error in dimension')

   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(),
                     precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(),
                     precision_backward, 'error on state (backward) ')
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
