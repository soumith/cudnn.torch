require 'cutorch'
require 'nn'
cudnn = require 'cudnn.env'
require('cudnn.ffi')
local C = cudnn.C
local ffi = require 'ffi'

--------------------------------------------------------------------
-- defaults, each should be overrideable via env var:
--------------------------------------------------------------------

cudnn.benchmark = false
cudnn.fastest = false

-- use new cudnn FindEx APIs
-- Warning: this option is experimental and assumes at least 2 warmup iterations!
cudnn.useFindEx = false

-- amount of memory to use on 1st iteration for FindEx
cudnn.initialWorkspaceBytes = 1024

--
cudnn.reservedGPUBytes = 1024*1024

cudnn.maxWorkspaceGPUMemPercent = 95

local maxStreamsPerDevice = 1024

--------------------------------------------------------------------
-- end defaults
--------------------------------------------------------------------

local numDevices = cutorch.getDeviceCount()
-- this tensor keeps track of whether a handle has been initialized or not
local handleStatus = torch.ByteTensor(numDevices,
                                  maxStreamsPerDevice):zero()
-- here we create an array of cudnn handle structs
cudnn.handle = ffi.new('struct cudnnContext*[?]', numDevices*maxStreamsPerDevice)
local function destroy(handle)
    local currentDevice = cutorch.getDevice()
    for i=1,numDevices do
        cutorch.setDevice(i)
        -- streams go from 0 to maxStreamsPerDevice - 1
        for j=0,maxStreamsPerDevice - 1 do
            if handleStatus[i][j + 1] == 1 then -- if handle was created
                cudnn.errcheck('cudnnDestroy', handle[(((i-1)*maxStreamsPerDevice) + j)]);
            end
        end
    end
    cutorch.setDevice(currentDevice)
end
ffi.gc(cudnn.handle, destroy)

cudnn.typemap = {
   ['torch.CudaHalfTensor']   = 'CUDNN_DATA_HALF',
   ['torch.CudaTensor']       = 'CUDNN_DATA_FLOAT',
   ['torch.CudaDoubleTensor'] = 'CUDNN_DATA_DOUBLE',
}

local sizeofmap = {
   ['torch.CudaHalfTensor']   = cutorch.hasHalf and ffi.sizeof('half') or 2,
   ['torch.CudaTensor']       = ffi.sizeof('float'),
   ['torch.CudaDoubleTensor'] = ffi.sizeof('double'),
}

function cudnn.sizeof(t)
   return sizeofmap[torch.type(t)]
end

local onemap = {
   ['torch.CudaHalfTensor']   = torch.FloatTensor({1}),
   ['torch.CudaTensor']       = torch.FloatTensor({1}),
   ['torch.CudaDoubleTensor'] = torch.DoubleTensor({1}),
}
local zeromap = {
   ['torch.CudaHalfTensor']   = torch.FloatTensor({0}),
   ['torch.CudaTensor']       = torch.FloatTensor({0}),
   ['torch.CudaDoubleTensor'] = torch.DoubleTensor({0}),
}
function cudnn.scalar(t, val)
   if val == 1 then
      return onemap[torch.type(t)]:data()
   elseif val == 0 then
      return zeromap[torch.type(t)]:data()
   else
      error('unknown scalar')
   end
end

-- TODO: determine if device supports true half and use true half on it
-- so far use float for half and float, double for double
local function determineHalfCapability(dev)
   local prop = cutorch.getDeviceProperties(dev)
   if prop.major >= 6 or prop.name:find'X1' then
      return 'CUDNN_DATA_HALF'
   else
      return 'CUDNN_DATA_FLOAT'
   end
end

local configmaps = {}
for i=1,cutorch.getDeviceCount() do
   configmaps[i] = {
      ['torch.CudaHalfTensor']   = determineHalfCapability(i),
      ['torch.CudaTensor']       = 'CUDNN_DATA_FLOAT',
      ['torch.CudaDoubleTensor'] = 'CUDNN_DATA_DOUBLE',
   }
end

cudnn.configmap = function(tensortype)
   return configmaps[cutorch.getDevice()][tensortype]
end

function cudnn.getHandle()
    local device = cutorch.getDevice()
    local stream = cutorch.getStream() -- starts from 0
    assert(stream < maxStreamsPerDevice, 'cudnn bindings only support max of : '
               .. maxStreamsPerDevice .. ' streams per device')
    -- lazy initialization of handles
    if handleStatus[device][stream + 1] == 0 then
        local status = C['cudnnCreate'](cudnn.handle
                                        + (((device-1) * maxStreamsPerDevice)
                                                + stream))
        if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
            local str = ffi.string(C.cudnnGetErrorString(status))
            error('Error in CuDNN: ' .. str)
        end
        handleStatus[device][stream + 1] = 1 -- mark handle as initialized
    end
    return cudnn.handle[(((device-1)*maxStreamsPerDevice) + stream)]
end

function cudnn.call(f, ...)
    C.cudnnSetStream(cudnn.getHandle(),
                     ffi.C.THCState_getCurrentStream(cutorch.getState()))
    return C[f](...)
end

local errcheck = function(f, ...)
   local status = cudnn.call(f, ...)
   if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
      local str = ffi.string(C.cudnnGetErrorString(status))
      error('Error in CuDNN: ' .. str .. ' ('..f..')')
      return false
   end
   return true
end
cudnn.errcheck = errcheck

function cudnn.toDescriptor(t)
   local typename = torch.typename(t)
   assert(cudnn.typemap[typename])
   local descriptor = ffi.new('struct cudnnTensorStruct*[1]')
   -- create descriptor
   errcheck('cudnnCreateTensorDescriptor', descriptor)
   -- set gc hook
   local function destroy(d)
      errcheck('cudnnDestroyTensorDescriptor', d[0]);
   end
   ffi.gc(descriptor, destroy)
   -- view 2D and 3D as 4D
   if t:dim() == 2 then
      t = t:view(t:size(1), t:size(2), 1, 1)
   elseif t:dim() == 3 then
      t = t:view(t:size(1), t:size(2), t:size(3), 1)
   end
   -- set descriptor
   local size = torch.LongTensor(t:size()):int()
   local stride = torch.LongTensor(t:stride()):int()

   errcheck('cudnnSetTensorNdDescriptor', descriptor[0], cudnn.typemap[typename],
            t:dim(), size:data(), stride:data())
   return descriptor
end

function cudnn.createDescriptors(count, descs_type, create_func, destroy_func)
   local ds = ffi.new(descs_type, count)
   for i = 0, count - 1 do
      errcheck(create_func, ds + i)
   end
   local function destroyDescriptors(ds)
      for i = 0, count - 1 do
         errcheck(destroy_func, ds[i])
      end
   end
   ffi.gc(ds, destroyDescriptors)
   return ds
end

local sharedBuffer = {}
local nextBufferSize = {}

local function setNextSize(buf, size, ifGreater)
   if size > buf.nextSize or not ifGreater then
      buf.nextSize = size
   end
end

-- may reassign currentSize
local function allocateStorage(buf, ifGreater)

   if buf.nextSize < 0 then
      buf.nextSize = buf.currentSize
   end

   local elSize = 8
   -- get number of elements in the buf, rounded up
   local newelem = math.floor((buf.nextSize+elSize-1)/elSize)

   if buf.storage then
      if (newelem == buf.storage:size()) or (ifGreater and newelem < buf.storage:size()) then
      else
         if cudnn.verbose then
            print( "allocateStorage: new WS size is ", buf.nextSize)
         end
         -- resize to just to make sure we return memory
         buf.storage:resize(0)
         buf.storage:resize(newelem)
      end
   else
      -- this is to be replaced with new cutorch tempbuf stuff
      -- may reassign currentSize again
      buf.storage = torch.CudaDoubleStorage(newelem)
   end

   buf.currentSize = buf.storage:size()*elSize
   buf.data = buf.storage:data()
   buf.nextSize = -1
end

local function sharedBufForCurrentStream()
    local device = cutorch.getDevice()
    local stream = cutorch.getStream() -- starts from 0
    if not sharedBuffer[device] then sharedBuffer[device] = {} end
    local buf = sharedBuffer[device][stream]
    if not buf then
       buf = {
          currentSize = cudnn.initialWorkspaceBytes,
          nextSize = -1
       }
       allocateStorage(buf)
       sharedBuffer[device][stream] = buf
    end
    return buf
end

function cudnn.getSharedWorkspace()
   local buf = sharedBufForCurrentStream()
   return buf.data, buf.currentSize
end

-- Creates a clone of luaStr that can be used to prevent side
-- effects when passing char* to C functions.
function cudnn.externalizeString(luaStr)
    local cStr = ffi.new("char[?]", #luaStr+1)
    ffi.copy(cStr, luaStr)
    return cStr
end

function cudnn.adjustSharedWorkspaceSize(bytesDelta)
   local buf = sharedBufForCurrentStream()
   setNextSize(buf, buf.currentSize + bytesDelta)
   allocateStorage(buf)
end

function cudnn.setSharedWorkspaceSize(bytes, ifGreater)
   local buf = sharedBufForCurrentStream()
   ifGreater = ifGreater or false
   bytes = bytes or cudnn.initialWorkspaceBytes
   setNextSize(buf, bytes, ifGreater)
   allocateStorage(buf, ifGreater)
end

local find = require('cudnn.find')

require('cudnn.SpatialConvolution')
require('cudnn.VolumetricConvolution')
require('cudnn.SpatialFullConvolution')
require('cudnn.Pooling')
require('cudnn.SpatialMaxPooling')
require('cudnn.SpatialAveragePooling')
require('cudnn.Pooling3D')
require('cudnn.VolumetricMaxPooling')
require('cudnn.VolumetricAveragePooling')
require('cudnn.Pointwise')
require('cudnn.ReLU')
require('cudnn.ClippedReLU')
require('cudnn.Tanh')
require('cudnn.Sigmoid')
require('cudnn.SpatialSoftMax')
require('cudnn.SpatialLogSoftMax')
require('cudnn.VolumetricSoftMax')
require('cudnn.VolumetricLogSoftMax')
require('cudnn.SoftMax')
require('cudnn.LogSoftMax')
require('cudnn.SpatialCrossMapLRN')
require('cudnn.BatchNormalization')
require('cudnn.SpatialBatchNormalization')
require('cudnn.VolumetricBatchNormalization')
require('cudnn.SpatialCrossEntropyCriterion')
require('cudnn.VolumetricCrossEntropyCriterion')
require('cudnn.TemporalConvolution')
require('cudnn.RNN')
require('cudnn.RNNTanh')
require('cudnn.RNNReLU')
require('cudnn.BLSTM')
require('cudnn.LSTM')
require('cudnn.BGRU')
require('cudnn.GRU')
require('cudnn.functional')
require('cudnn.convert')

function cudnn.reset()
-- this resets everything
   if cudnn.verbose then
      print("cudnn::reset for device #", cutorch.getDevice())
   end
   cutorch.synchronize()
   -- make sure shared buffers that may have been cached, have 0 size
   for i=1,numDevices do
      sharedBuffer[i] = {}
   end
   collectgarbage()
   -- this resets internal algorithm finder state machine and cache
   find.reset()
end

return cudnn
