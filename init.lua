require 'cutorch'
require 'nn'
cudnn = {}
include 'ffi.lua'
local C = cudnn.C
local ffi = require 'ffi'

local errcheck = function(f, ...)
   local status = C[f](...)
   if status ~= 'CUDNN_STATUS_SUCCESS' then
      print("Error in CuDNN. Status Code: ", tonumber(status))
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
   assert(t:dim() == 4);
   assert(torch.typename(t) == 'torch.CudaTensor')
   local descriptor = ffi.new('struct cudnnTensor4dStruct*[1]')
   -- create descriptor
   errcheck('cudnnCreateTensor4dDescriptor', descriptor)
   -- set gc hook
   local function destroy(d) 
      errcheck('cudnnDestroyTensor4dDescriptor', d[0]); 
   end
   ffi.gc(descriptor, destroy)
   -- set descriptor
   errcheck('cudnnSetTensor4dDescriptorEx', descriptor[0], 'CUDNN_DATA_FLOAT', 
            t:size(1), t:size(2), t:size(3), t:size(4),
            t:stride(1), t:stride(2), t:stride(3), t:stride(4))
   return descriptor
end

include 'SpatialConvolution.lua'
include 'SpatialMaxPooling.lua'
include 'ReLU.lua'
include 'Tanh.lua'
include 'Sigmoid.lua'
--[[ 
include 'Softmax.lua'
]]--
