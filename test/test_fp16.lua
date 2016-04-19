require 'cudnn'
require 'cunn'


x = torch.randn(1,3,25,27)
m = cudnn.SpatialConvolution(3,16,7,7,1,1,3,3)

function fwd(datatype)
   x = x:type(datatype)
   m:clearDesc()
   m = m:type(datatype)
   return m:forward(x):clone()
end

print('=========== single ===========')
y1 = fwd('torch.CudaTensor')
print(y1:sub(1,1, 1,3, 1,3, 1,3))

print('============ half ============')
y2 = fwd('torch.CudaHalfTensor'):cuda()
print(y2:sub(1,1, 1,3, 1,3, 1,3))

print('============ diff ============')
yDiff = y2-y1
print(yDiff:sub(1,1, 1,3, 1,3, 1,3))
