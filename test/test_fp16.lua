require 'cudnn'
require 'cunn'

-- make 'view' work
x = torch.randn(1,1,3,4):cudaHalf()
print(x)
print(x:view(torch.LongStorage{1,1,4,3}))


x = torch.randn(1,3,25,27)
m = cudnn.SpatialConvolution(3,16,7,7,1,1,3,3)

datatype = 'torch.CudaTensor'
--datatype = 'torch.CudaHalfTensor'
x = x:type(datatype)
m:clearDesc()
m = m:type(datatype)
y = m:forward(x):clone()
y = y:cuda()
print(y:sub(1,1, 1,3, 1,3, 1,3))

