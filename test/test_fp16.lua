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

m = cudnn.SpatialMaxPooling

function fwdReLU(x, datatype)
   x = x:clone():type(datatype)
   m = m:type(datatype)
   return m:forward(x):clone()
end
eval(torch.randn(1,3,25,27), fwdReLU)

