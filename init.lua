require 'cutorch'
require 'nn'
cudnn = require 'cudnn.env'
require('cudnn.ffi')
local C = cudnn.C
local ffi = require 'ffi'

local thc = ffi.C
if ffi.os == "Windows" then thc = ffi.load("THC") end

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


local function fasterHalfMathTypeForCurrentDevice()
   -- get info from cutorc
   if cutorch.hasFastHalfInstructions() then
      return 'CUDNN_DATA_HALF'
   else
      return 'CUDNN_DATA_FLOAT'
   end
end

local configMaths = {}

local function configureMath(overrides)
   local currentDevice = cutorch.getDevice()
   for i=1,cutorch.getDeviceCount() do
      cutorch.setDevice(i)
      configMaths[i] = {
         ['torch.CudaHalfTensor']   = fasterHalfMathTypeForCurrentDevice(),
         ['torch.CudaTensor']       = 'CUDNN_DATA_FLOAT',
         ['torch.CudaDoubleTensor'] = 'CUDNN_DATA_DOUBLE',
      }
      -- apply overrides
      if overrides then
         for k,v in pairs(overrides) do configMaths[i][k] = v end
      end
   end
   cutorch.setDevice(currentDevice)
end
cudnn.configureMath = configureMath

-- TODO: rename to something like "configuredMathType" on next refactor
-- also, should move torch.type() inside
cudnn.configmap = function(tensortype)
   return configMaths[cutorch.getDevice()][tensortype]
end

configureMath()

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
                     thc.THCState_getCurrentStream(cutorch.getState()))
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



function cudnn.setConvolutionDescriptor(data, desc)
   if not data.arrayLength then data.arrayLength = #data.padA end
   if not data.upscaleA then data.upscaleA =  torch.IntStorage(data.arrayLength):fill(1) end
   if not data.mode then data.mode = 'CUDNN_CROSS_CORRELATION' end

   local myDesc = desc or cudnn.createDescriptors(
      1, 'struct cudnnConvolutionStruct*[?]',
      'cudnnCreateConvolutionDescriptor', 'cudnnDestroyConvolutionDescriptor')
   errcheck('cudnnSetConvolutionNdDescriptor', myDesc[0],
            data.arrayLength,
            torch.IntTensor(data.padA):data(),
            torch.IntTensor(data.filterStrideA):data(),
            torch.IntTensor(data.upscaleA):data(),
            data.mode,
            data.dataType)
   return myDesc
end

function cudnn.setFilterDescriptor(data, filterDesc)
   local myDesc = filterDesc or cudnn.createDescriptors(
      1, 'struct cudnnFilterStruct*[?]',
      'cudnnCreateFilterDescriptor', 'cudnnDestroyFilterDescriptor')
   local dims = data.nbDims or #data.filterDimA
   errcheck('cudnnSetFilterNdDescriptor', myDesc[0],
            data.dataType, data.format or 'CUDNN_TENSOR_NCHW',
            dims, torch.IntTensor(data.filterDimA):data());
   return myDesc
end

local sharedBuffer = {}
local nextBufferSize = {}

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

local function sharedBufForStream(device, stream)
   device = device or cutorch.getDevice()
   stream = stream or cutorch.getStream() -- starts from 0
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

function cudnn.getSharedWorkspace(device, stream)
   device = device or cutorch.getDevice()
   stream = stream or cutorch.getStream()
   local buf = sharedBufForStream(device, stream)
   return buf.data, buf.currentSize
end

-- Creates a clone of luaStr that can be used to prevent side
-- effects when passing char* to C functions.
function cudnn.externalizeString(luaStr)
    local cStr = ffi.new("char[?]", #luaStr+1)
    ffi.copy(cStr, luaStr)
    return cStr
end

function cudnn.adjustSharedWorkspaceSize(bytesDelta, device, stream)
   local buf = sharedBufForStream(device, stream)
   buf.nextSize = buf.currentSize + bytesDelta
   allocateStorage(buf)
end

function cudnn.setNextWorkspaceSize(bytes, device, stream)
   local buf = sharedBufForStream(device, stream)
   buf.nextSize = bytes
   return buf
end

function cudnn.setSharedWorkspaceSize(bytes, ifGreater, device, stream)
   bytes = bytes or cudnn.initialWorkspaceBytes
   local buf = cudnn.setNextWorkspaceSize(bytes, device, stream)
   allocateStorage(buf, ifGreater)
end

cudnn.find = require('cudnn.find')

require('cudnn.SpatialConvolution')
require('cudnn.VolumetricConvolution')
require('cudnn.SpatialFullConvolution')
require('cudnn.VolumetricFullConvolution')
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
   cudnn.find.reset()
end

return cudnn
