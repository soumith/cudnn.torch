require 'cudnn'
require 'cunn'


function eval(x, fwd)
   print('=========== input ============')
   print(x:sub(1,1, 1,3, 1,3, 1,3))

   print('=========== single ===========')
   y1 = fwd(x:clone(), 'torch.CudaTensor')
   print(y1:sub(1,1, 1,3, 1,3, 1,3))
   
   print('============ half ============')
   y2 = fwd(x:clone(), 'torch.CudaHalfTensor'):cuda()
   print(y2:sub(1,1, 1,3, 1,3, 1,3))
   
   print('============ diff ============')
   yDiff = y2-y1
   print(yDiff:sub(1,1, 1,3, 1,3, 1,3))
end





print('==============================')
print('===== SpatialConvolution =====')
print('==============================')
print('')

m = cudnn.SpatialConvolution(3,16,7,7,1,1,3,3)

function fwdSpatialConv(x, datatype)
   x = x:type(datatype)
   --m:clearDesc()
   m = m:type(datatype)
   return m:forward(x):clone()
end
eval(torch.randn(1,3,25,27), fwdSpatialConv)


print('==============================')
print('============ ReLU ============')
print('==============================')
print('')

m = cudnn.ReLU(true)

function fwdReLU(x, datatype)
   x = x:clone():type(datatype)
   m = m:type(datatype)
   return m:forward(x):clone()
end
eval(torch.randn(1,3,25,27), fwdReLU)


print('==============================')
print('========== MaxPool ===========')
print('==============================')
print('')

m = cudnn.SpatialMaxPooling(2,2)

function fwdMaxPool(x, datatype)
   x = x:clone():type(datatype)
   m = m:type(datatype)
   return m:forward(x):clone()
end
eval(torch.randn(1,3,25,27), fwdMaxPool)


print('==============================')
print('========== SoftMax ===========')
print('==============================')
print('')

m = cudnn.SpatialSoftMax()

function fwdSoftMax(x, datatype)
   x = x:clone():type(datatype)
   m = m:type(datatype)
   return m:forward(x):clone()
end
eval(torch.randn(1,3,25,27), fwdSoftMax)


-- benchmarking SpatialConvolution
h = 60
w = 80
nInputC = 64
nOutputC = 256
k = 7
numRuns = 5000
numOps = 2*nOutputC*nInputC*k*k*(h-k+1)*(w-k+1)
datatype = 'torch.CudaTensor'
datatype = 'torch.CudaHalfTensor'
--cutorch.setDevice(2)

i1 = torch.randn(1,nInputC,h,w):type(datatype)
m1 = cudnn.SpatialConvolution(nInputC,nOutputC, k,k):fastest():type(datatype)
for i = 1,200 do
   o1 = m1:forward(i1)
end
cutorch.synchronize()
t1 = torch.Timer()
for i = 1,numRuns do
   o1 = m1:forward(i1)
end
cutorch.synchronize()
timePerPass = t1:time().real/numRuns
print(datatype .. ': ', nInputC, nOutputC, kH, kW, iH, iW, nBatch, timePerPass, numOps/1e9/timePerPass)
