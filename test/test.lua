require 'cudnn'
require 'cunn'

local cudnntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local precision_jac = 1e-3
local nloop = 1
local times = {}
local mytester

function cudnntest.SpatialConvolution_forward_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cuda()
   local groundtruth = sconv:forward(input)
   cutorch.synchronize()
   local gconv = cudnn.SpatialConvolution(from,to,ki,kj,si,sj):cuda():fastest()
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
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local scale = math.random()

   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,to,outj,outi):cuda()
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cuda()
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput, scale)
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cudnn.SpatialConvolution(from,to,ki,kj,si,sj):cuda():fastest()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput, scale)
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
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
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
   mytester:assertlt(error:abs():max(), precision_forward,
                     'error on state (forward) ')
end


function cudnntest.SpatialConvolution_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
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

   mytester:assertlt(error:abs():max(), precision_backward,
                     'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward,
                     'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward,
                     'error on bias (backward) ')
end

function cudnntest.VolumetricConvolution_forward_single()
   local from = math.random(1,16)
   local to = math.random(1,16)
   local ki = math.random(3,5)
   local kj = math.random(3,5)
   local kk = math.random(3,5)
   local si = math.random(1,ki-1)
   local sj = math.random(1,kj-1)
   local sk = math.random(1,kk-1)
   local outi = math.random(1,17)
   local outj = math.random(1,17)
   local outk = math.random(1,5)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk
   local input = torch.randn(from,ink,inj,ini):cuda()
   local sconv = nn.VolumetricConvolution(from,to,kk,ki,kj,sk,si,sj):float()
   local groundtruth = sconv:forward(input:float())
   cutorch.synchronize()
   local gconv = cudnn.VolumetricConvolution(from,to,kk,ki,kj,sk,si,sj):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   local rescuda = gconv:forward(input)
   cutorch.synchronize()
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward,
                     'error on state (forward) ')
end

function cudnntest.VolumetricConvolution_backward_single()
   local from = math.random(1,16)
   local to = math.random(1,16)
   local ki = math.random(3,5)
   local kj = math.random(3,5)
   local kk = math.random(3,5)
   local si = math.random(1,ki-1)
   local sj = math.random(1,kj-1)
   local sk = math.random(1,kk-1)
   local outi = math.random(1,17)
   local outj = math.random(1,17)
   local outk = math.random(1,5)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk
   local input = torch.randn(from,ink,inj,ini):cuda()
   local gradOutput = torch.randn(to,outk,outj,outi):cuda()
   local sconv = nn.VolumetricConvolution(from,to,kk,ki,kj,sk,si,sj):float()
   sconv:forward(input:float())
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input:float(), gradOutput:float())
   cutorch.synchronize()
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   local gconv = cudnn.VolumetricConvolution(from,to,kk,ki,kj,sk,si,sj):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(input)
   cutorch.synchronize()

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   cutorch.synchronize()

   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias

   local error = rescuda:float() - groundgrad:float()
   local werror = weightcuda:float() - groundweight:float()
   local berror = biascuda:float() - groundbias:float()

   mytester:assertlt(error:abs():max(), precision_backward,
                     'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward,
                     'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward,
                     'error on bias (backward) ')

end

function cudnntest.VolumetricMaxPooling_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local kk = math.random(2,4)
   local si = ki
   local sj = kj
   local sk = kk
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local outk = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk
   local input = torch.randn(bs,from,ink,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outk,outj,outi):cuda()

   local sconv = nn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj):float()
   local groundtruth = sconv:forward(input:float())
   local groundgrad = sconv:backward(input:float(), gradOutput:float())
   cutorch.synchronize()
   local gconv = cudnn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj):cuda()
   local rescuda = gconv:forward(input)
   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')
   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 5, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 5, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cudnntest.VolumetricMaxPooling_single()
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local kk = math.random(2,4)
   local si = ki
   local sj = kj
   local sk = kk
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local outk = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk
   local input = torch.randn(from,ink,inj,ini):cuda()
   local gradOutput = torch.randn(from,outk,outj,outi):cuda()

   local sconv = nn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj):float()
   local groundtruth = sconv:forward(input:float())
   local groundgrad = sconv:backward(input:float(), gradOutput:float())
   cutorch.synchronize()
   local gconv = cudnn.VolumetricMaxPooling(kk,ki,kj,sk,si,sj):cuda()
   local _ = gconv:forward(input)
   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')
   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward,
                     'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward,
                     'error on state (backward) ')
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
   local _ = gconv:forward(input)
   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')
   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 3, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward,
                     'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward,
                     'error on state (backward) ')
end

function cudnntest.SpatialAveragePooling_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   local groundtruth = sconv:forward(input):clone()
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.SpatialAveragePooling(ki,kj,si,sj):cuda()
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

function cudnntest.SpatialAveragePooling_single()
   local from = math.random(1,32)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(from,outj,outi):cuda()

   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   local groundtruth = sconv:forward(input):clone()
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   local _ = gconv:forward(input)
   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')
   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 3, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward,
                     'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward,
                     'error on state (backward) ')
end

local function nonlinSingle(nonlin)
   local from = math.random(1,32)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(from,inj,ini):cuda()
   local gradOutput = torch.randn(from,outj,outi):cuda()

   local sconv = nn[nonlin]():cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   -- 50% prob to choose inplace or out-of-place
   local inplace = false
   if math.random(0,1) == 1 then
      inplace = true
   end
   local gconv = cudnn[nonlin](inplace):cuda()
   local input__ = input:clone()
   local _ = gconv:forward(input__)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local input__ = input:clone()
   local gradOutput__ = gradOutput:clone()
   local rescuda = gconv:forward(input__)
   local resgrad = gconv:backward(input__, gradOutput__)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 3, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 3, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward,
                     'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward,
                     'error on state (backward) ')
end

function nonlinBatch(nonlin)
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = outi
   local inj = outj
   local input = torch.randn(bs,from,inj,ini):cuda()
   local gradOutput = torch.randn(bs,from,outj,outi):cuda()

   local sconv = nn[nonlin]():cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   -- 50% prob to choose inplace or out-of-place
   local inplace = false
   if math.random(0,1) == 1 then
      inplace = true
   end
   local gconv = cudnn[nonlin](inplace):cuda()
   local input__ = input:clone()
   local rescuda = gconv:forward(input__)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local input__ = input:clone()
   local gradOutput__ = gradOutput:clone()
   local rescuda = gconv:forward(input__)
   local resgrad = gconv:backward(input__, gradOutput__)
   cutorch.synchronize()
   mytester:asserteq(rescuda:dim(), 4, 'error in dimension')
   mytester:asserteq(resgrad:dim(), 4, 'error in dimension')
   local error = rescuda:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(), precision_forward,
                     'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   mytester:assertlt(error:abs():max(), precision_backward,
                     'error on state (backward) ')
end

function cudnntest.ReLU_single()
   nonlinSingle('ReLU')
end

function cudnntest.ReLU_batch()
   nonlinBatch('ReLU')
end

function cudnntest.Tanh_single()
   nonlinSingle('Tanh')
end

function cudnntest.Tanh_batch()
   nonlinBatch('Tanh')
end

function cudnntest.Sigmoid_single()
   nonlinSingle('Sigmoid')
end

function cudnntest.Sigmoid_batch()
   nonlinBatch('Sigmoid')
end

function cudnntest.SoftMax_single()
   local sz = math.random(1,64)
   local input = torch.randn(sz):cuda()
   local gradOutput = torch.randn(sz):cuda()

   local sconv = nn.SoftMax():cuda()
   local groundtruth = sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   cutorch.synchronize()
   local gconv = cudnn.SoftMax():cuda()
   local _ = gconv:forward(input)

   -- serialize and deserialize
   torch.save('modelTemp.t7', gconv)
   gconv = torch.load('modelTemp.t7')

   local rescuda = gconv:forward(input)
   local resgrad = gconv:backward(input, gradOutput)
   cutorch.synchronize()
   local error = rescuda:float() - groundtruth:float()
   local errmax = error:abs():max()
   if (errmax ~= errmax) then
      local state = {}
      state.input = input
      state.gradOutput = gradOutput
      state.rescuda = rescuda
      state.resgrad = resgrad
      state.groundtruth = groundtruth
      state.groundgrad = groundgrad
      print(#input)
      torch.save('badSoftMax.t7', state)
      print(#input)
   end
   mytester:assertlt(errmax, precision_forward,
                     'error on state (forward) ')
   error = resgrad:float() - groundgrad:float()
   errmax = error:abs():max()
   if (errmax ~= errmax) then
      local state = {}
      state.input = input
      state.gradOutput = gradOutput
      state.rescuda = rescuda
      state.resgrad = resgrad
      state.groundtruth = groundtruth
      state.groundgrad = groundgrad
      print(#input)
      torch.save('badSoftMax.t7', state)
      print(#input)
   end
   mytester:assertlt(errmax, precision_backward,
                     'error on state (backward) ')
end

function cudnntest.SoftMax_batch()
   local bs = math.random(1,32)
   local from = math.random(1,32)
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

function cudnntest.functional_SpatialBias()
   local bs = math.random(1,32)
   local from = math.random(1,32)
   local to = math.random(1,64)
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local scale = torch.uniform()
   local input = torch.zeros(bs,from,inj,ini):cuda()
   local mod = cudnn.SpatialConvolution(from,to,ki,kj,si,sj):cuda()
   mod.weight:zero()
   local groundtruth = mod:forward(input)
   local result = groundtruth:clone():zero()
   cudnn.functional.SpatialBias_updateOutput(mod.bias, result)
   local error = result:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(),
                     precision_forward, 'error on forward ')

   mod:zeroGradParameters()
   local gradOutput = groundtruth:clone():normal()
   mod:backward(input, gradOutput, scale)
   local groundtruth = mod.gradBias
   local result = groundtruth:clone():zero()
   cudnn.functional.SpatialBias_accGradParameters(gradOutput, result, scale)
   error = result:float() - groundtruth:float()
   mytester:assertlt(error:abs():max(),
                     precision_backward, 'error on accGradParameters ')
end


torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(cudnntest)

for i=1,cutorch.getDeviceCount() do
   print('Running test on device: ' .. i)
   cutorch.setDevice(i)
   mytester:run()
end

os.execute('rm -f modelTemp.t7')
