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

cudnn.handle = ffi.new('struct cudnnContext*[1]')
-- create handle
errcheck('cudnnCreate', cudnn.handle)
local function destroy(handle) 
   errcheck('cudnnDestroy', handle[0]); 
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
      errcheck('cudnnDestroyTensor4dDescriptor', descriptor[0]); 
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
