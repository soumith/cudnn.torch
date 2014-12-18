require 'cutorch'
require 'nn'
cudnn = {}
include 'ffi.lua'
local C = cudnn.C
local ffi = require 'ffi'

local errcheck = function(f, ...)
   local status = C[f](...)
   if status ~= 'CUDNN_STATUS_SUCCESS' then
      local str = ffi.string(C.cudnnGetErrorString(status))
      error('Error in CuDNN: ' .. str)
   end
end
cudnn.errcheck = errcheck

local numDevices = cutorch.getDeviceCount()
local currentDevice = cutorch.getDevice()
cudnn.handle = ffi.new('struct cudnnContext*[?]', numDevices)
-- create handle
for i=1,numDevices do
   cutorch.setDevice(i)
   errcheck('cudnnCreate', cudnn.handle+i-1)
end
cutorch.setDevice(currentDevice)

local function destroy(handle)
   local currentDevice = cutorch.getDevice()
   for i=1,numDevices do
      cutorch.setDevice(i)
      errcheck('cudnnDestroy', handle[i-1]);
   end
   cutorch.setDevice(currentDevice)
end
ffi.gc(cudnn.handle, destroy)

function cudnn.toDescriptor(t)
   assert(torch.typename(t) == 'torch.CudaTensor')
   local descriptor = ffi.new('struct cudnnTensorStruct*[1]')
   -- create descriptor
   errcheck('cudnnCreateTensorDescriptor', descriptor)
   -- set gc hook
   local function destroy(d)
      errcheck('cudnnDestroyTensorDescriptor', d[0]);
   end
   ffi.gc(descriptor, destroy)
   -- set descriptor
   local size = torch.LongTensor(t:size()):int()
   local stride = torch.LongTensor(t:stride()):int()
   errcheck('cudnnSetTensorNdDescriptor', descriptor[0], 'CUDNN_DATA_FLOAT',
            t:dim(), size:data(), stride:data())
   return descriptor
end

include 'SpatialConvolution.lua'
include 'Pooling.lua'
include 'SpatialMaxPooling.lua'
include 'SpatialAveragePooling.lua'
include 'Pointwise.lua'
include 'ReLU.lua'
include 'Tanh.lua'
include 'Sigmoid.lua'
include 'SpatialSoftMax.lua'
include 'SoftMax.lua'
