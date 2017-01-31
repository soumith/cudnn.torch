local ffi = require 'ffi'

local find = {}
find.__index = find

-- default is to get verbose on errors
find.verbose=false
find.verboseError=true
find.verboseFallback=true

-- constants to index array tables below
local Fwd, BwdFilter, BwdData = 1, 2, 3

-- constants to select algo family, to index algoFamilies
local GetFamily, FindFamily, FindExFamily = 1,2,3

local warmupIterations = 0

local Meg = 1024*1024

-- cudnnGetxxx APIs: default, when cudnn.benchmark == false
local getAlgos = {'cudnnGetConvolutionForwardAlgorithm',
                  'cudnnGetConvolutionBackwardFilterAlgorithm',
                  'cudnnGetConvolutionBackwardDataAlgorithm'}
local getWSAlgos = {'cudnnGetConvolutionForwardWorkspaceSize',
                    'cudnnGetConvolutionBackwardFilterWorkspaceSize',
                    'cudnnGetConvolutionBackwardDataWorkspaceSize'}

-- cudnnFindxxx APIs: default, when cudnn.benchmark == true
local findAlgos = {'cudnnFindConvolutionForwardAlgorithm',
                   'cudnnFindConvolutionBackwardFilterAlgorithm',
                   'cudnnFindConvolutionBackwardDataAlgorithm'}

-- cudnnFindxxxEx APIs: default, when cudnn.benchmark == true and cudnn.useFindEx == true
local findExAlgos = {'cudnnFindConvolutionForwardAlgorithmEx',
                     'cudnnFindConvolutionBackwardFilterAlgorithmEx',
                     'cudnnFindConvolutionBackwardDataAlgorithmEx'}

local algoFamilies = { getAlgos, findAlgos, findExAlgos}

local fwdAlgoNames = {
   "IMPLICIT_GEMM",
   "IMPLICIT_PRECOMP_GEMM",
   "GEMM",
   "DIRECT",
   "FFT",
   "FFT_TILING",
   "WINOGRAD",
   "WINOGRAD_NONFUSED"
}

local bwdFilterAlgoNames = {
   "ALGO_0",
   "ALGO_1",
   "FFT",
   "ALGO_3",
   "WINOGRAD",
   "WINOGRAD_NONFUSED"
}

local bwdDataAlgoNames = {
   "ALGO_0",
   "ALGO_1",
   "FFT",
   "FFT_TILING",
   "WINOGRAD",
   "WINOGRAD_NONFUSED"
}

local algoNames = {fwdAlgoNames, bwdFilterAlgoNames, bwdDataAlgoNames}

local function convDataString(layer)
   local info = ''
   if layer.convDescData then
      local desc = layer.convDescData
      info = ' convDesc=[mode : ' .. desc.mode .. ' datatype : ' .. desc.dataType .. ']'
   end
   return info .. ' hash=' ..  layer.autotunerHash
end

local function verboseCall(layer, f, ...)
   local status = cudnn.call(f, ...)
   if (status ~= ffi.C.CUDNN_STATUS_SUCCESS) and (find.verbose or find.verboseError) then
      print("\n" .. f .. " failed: ", tonumber(status), convDataString(layer))
   end
   return status
end
find.verboseCall = verboseCall

local function checkedCall(layer, f, ...)
   local status = verboseCall(layer, f, ...)
   if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
      local str = ffi.string(cudnn.C.cudnnGetErrorString(status))
      error('Error in CuDNN: ' .. str .. ' ('..f..')')
   end
   return status
end
find.checkedCall = checkedCall

local function noFallback(layer)
   if find.verbose or find.verboseFallback then
      print("\nfind.defaultFallback: verboseCall failed for:  ", convDataString(layer))
   end
   return false
end

local function fallbackWarning(layer, msg)
   if find.verbose or find.verboseFallback then
      print("\n *** find.verboseFallback: " .. msg ..
            "\n *** Falling back to 32-bit math for: " .. convDataString(layer))
      print(" *** [ Set cudnn.find.verboseFallback to false to disable this message ] *** ")
      print(" *** [ Alternatively, you may force CUDNN to always operate on CudaHalfTensors via 32-bit float conversion, in Lua: ] ***\n"
               .." *** [ cudnn.configureMath({ ['torch.CudaHalfTensor']   = 'CUDNN_DATA_FLOAT'} ] ***")
      print(" *** [ Note: result may be faster or slower than native FP16, depending on your GPU and CUDNN operations ] *** ")
   end
end

local function defaultFallback(layer, replay)
   -- read conv descriptor
   local convDescData = layer.convDescData
   if convDescData and convDescData.dataType == "CUDNN_DATA_HALF" then
      fallbackWarning(layer, replay
                         and "16->32 bit fallback replay "
                         or "No native FP16 algo found, will try 32-bit math")
      -- update our record with fallback value
      convDescData.dataType = "CUDNN_DATA_FLOAT"
      -- update the descriptor in CUDNN
      cudnn.setConvolutionDescriptor(convDescData, layer.convDesc)
      return true
   else
      return false
   end
end

-- Find State and Cache (per device)
local function initState(id)
   local finder = {}
   setmetatable(finder,find)
   finder.id = id
   finder:resetAlgorithmCache()
   finder.iteration = 0
   if cutorch.hasHalf then
      finder.fallback = defaultFallback
   end
   return finder
end

local finders = nil
-- this resets algorithm cache for device

local function setAlgoFamily()
 return cudnn.benchmark
      and (cudnn.useFindEx and FindExFamily or FindFamily)
      or GetFamily
end

function find:resetAlgorithmCache()
   self.calculatedWorkspaceSize  = {}
   self:calculateMaxWorkspaceSize()
   self.algoFamily = setAlgoFamily()
   self.autotunerCache = {{}, {}, {}}
end

function find.reset(warmup)
   cutorch:synchronizeAll()
   finders = {}
   warmupIterations = warmup or 0
end

function find.get()
   local device = cutorch.getDevice()
   local it = finders[device]
   if not it then
      it = initState(device)
      finders[device] = it
   end
   return it
end

function find:lookup(layer, findAPI_idx)
   return  self.autotunerCache[findAPI_idx][layer.autotunerHash]
end

-- record algo, memory in cache
function find:store(layer, findAPI_idx, cachedAlgo)
   if warmupIterations==0 then
      self.autotunerCache[findAPI_idx][layer.autotunerHash] = cachedAlgo
   end
end

function find:calculateMaxWorkspaceSize(reserve, fraction)
   if not reserve or reserve < cudnn.reservedGPUBytes then reserve = cudnn.reservedGPUBytes end
   local max_fraction =  cudnn.maxWorkspaceGPUMemPercent/100
   if not fraction or fraction > max_fraction then fraction = max_fraction end
   local buf, curSize = cudnn.getSharedWorkspace()
   -- check current usage
   local freeMemory, totalMemory = cutorch.getMemoryUsage(self.id)
   local newSize= (freeMemory+curSize-reserve) * fraction
   self.maxWorkspaceSize = newSize
   if find.verbose then
      print("calculateMaxWorkspaceSize Memory: ", freeMemory/Meg, "M free, " , totalMemory/Meg, "M total, " , self.maxWorkspaceSize/Meg, "M Workspace" )
   end
end

function find:setCalculatedWorkspaceSize(greater)
   local device = cutorch.getDevice()
   for stream,bytes in pairs (self.calculatedWorkspaceSize) do
      cudnn.setSharedWorkspaceSize(bytes, greater, device, stream)
   end
end

function find:pickAlgoAndCalculateWorkspaceSize(cachedAlgo)
   local stream = cutorch.getStream()

   if not self.calculatedWorkspaceSize[stream] then
      self.calculatedWorkspaceSize[stream] = 0
   end

   if self.calculatedWorkspaceSize[stream] > self.maxWorkspaceSize then
      self.calculatedWorkspaceSize[stream] = self.maxWorkspaceSize
   end

   -- find algo with a size that keeps the sum of stream sizes within ws size
   for a=1,#cachedAlgo do
      local algoSize = cachedAlgo[a].memory
      local delta = algoSize - self.calculatedWorkspaceSize[stream]
      if delta > 0 then
         -- check if we still fit
         local totalWS = 0
         for s,sz in pairs(self.calculatedWorkspaceSize) do
            totalWS = totalWS + sz
         end
         if totalWS + delta < self.maxWorkspaceSize then
            self.calculatedWorkspaceSize[stream] = algoSize
            return a
         end
      else
         -- keep previously calculated WS size for the stream
         return a
      end  -- delta
   end
   return 0
end

function find:reserveBytes(layer)
   local reserve = cudnn.reservedGPUBytes
   -- todo: implement layer method returning memory allocation size
   reserve = reserve + 2*layer.weight:nElement()*layer.weight:elementSize()
   return reserve
end

function find:verifyReserveForWeights(layer)
   local freeMemory, totalMemory = cutorch.getMemoryUsage(self.id)
   local reserve =  self:reserveBytes(layer)
   if freeMemory < reserve then
      -- let's make sure we still have space to reallocate our data
      cudnn.adjustSharedWorkspaceSize(freeMemory - reserve)
   end
end


function find:checkIteration(layer, findAPI_idx)
   if warmupIterations == 0 then return end
   if not layer.iteration then layer.iteration = {0,0,0} end

   -- find last iteration
   local max_iter = 0
   for k,v in pairs(layer.iteration) do
      if v > max_iter then max_iter = v end
   end

   if (self.iteration < max_iter and max_iter > 1) then
      self.iteration = max_iter
      if find.verbose then  print ("CUDNN Find SM: iteration #", self.iteration) end
      if warmupIterations > 0 then warmupIterations = warmupIterations -1 end
  end
   layer.iteration[findAPI_idx] = layer.iteration[findAPI_idx] + 1
end

local cachedAlgo
local nAlgos = 10
-- pre-allocated parameters for the APIs: Fwd, Bwd and BwdD use all different enums
local perfResultsArray = { ffi.new('cudnnConvolutionFwdAlgoPerf_t[?]', nAlgos),
                           ffi.new('cudnnConvolutionBwdFilterAlgoPerf_t[?]', nAlgos),
                           ffi.new('cudnnConvolutionBwdDataAlgoPerf_t[?]', nAlgos) }
local numPerfResults = ffi.new('int[1]')
local algType = { ffi.new('cudnnConvolutionFwdAlgo_t[?]', 1),
                  ffi.new('cudnnConvolutionBwdFilterAlgo_t[?]', 1),
                  ffi.new('cudnnConvolutionBwdDataAlgo_t[?]', 1)}

function find:setupAlgo(layer, findAPI_idx, algSearchMode, params)
        local retAlgo
        local cacheHit = '[found in cache]'
        local useFallback = false

        -- Check if it's a new iteration, decrement warmup
        self:checkIteration(layer, findAPI_idx)

        local curWorkspace, curWorkspaceSize = cudnn.getSharedWorkspace()
        local validResults = 0
        local API = algoFamilies[self.algoFamily][findAPI_idx]
        local perfResults = perfResultsArray[findAPI_idx]
        -- try to find algo in the cache first
        cachedAlgo =  self:lookup(layer, findAPI_idx)

        if cachedAlgo then
           validResults = #cachedAlgo
           useFallback = cachedAlgo[1].fallback
           -- need to replay fallback on cache hit
           if useFallback then self.fallback(layer, true) end
        else
           cacheHit = ''
           cachedAlgo = {}
--algo family might have changed, reset it
           self.algoFamily = setAlgoFamily()
           local API = algoFamilies[self.algoFamily][findAPI_idx]
           if self.algoFamily == FindExFamily then
              -- clone output tensor
                 local paramstmp = params[7]
                 params[7] = paramstmp:clone()
              -- temporarily set WS size to the max
              self:calculateMaxWorkspaceSize()
              cudnn.setSharedWorkspaceSize(self.maxWorkspaceSize)
           else
              if self.algoFamily == FindFamily then
                 -- Find() APIs use free GPU memory to find algo, release our WS bytes
                 cudnn.setSharedWorkspaceSize(0)
              end
           end

           local function callCudnn(layer)
              local ret = 0
              validResults = 0
              if not layer.convDesc or not layer.convDesc[0] then
                 error("No convDesc set on layer!")
              end

              if self.algoFamily == FindExFamily then
                 -- query temp workspace size
                 local tempWorkspace, tempWorkspaceSize = cudnn.getSharedWorkspace()
                 ret =  verboseCall(layer, API,
                                    cudnn.getHandle(),
                                    params[1], params[2]:data(), params[3], params[4]:data(), layer.convDesc[0], params[6], params[7]:data(),
                                    nAlgos, numPerfResults, perfResults, tempWorkspace, tempWorkspaceSize)
                 params[7]=paramstmp
              else
                 if self.algoFamily == FindFamily then
                    ret = verboseCall(layer, API,
                                      cudnn.getHandle(),
                                      params[1], params[3], layer.convDesc[0], params[6],
                                      nAlgos, numPerfResults, perfResults)
                 else
                    -- GetFamily: emulate findXXX results layout
                    numPerfResults[0]=1
                    perfResults[0].algo = 0
                    perfResults[0].memory = 0
                    perfResults[0].status = 1

                    local algWorkspaceLimit = layer.workspace_limit
                       or (layer.nInputPlane * layer.kH * layer.kW * layer.weight.elementSize())

                    ret = cudnn.call(API,
                                     cudnn.getHandle(),
                                     params[1], params[3], layer.convDesc[0], params[6],
                                     algSearchMode, algWorkspaceLimit, algType[findAPI_idx])
                    if ret ~= 0 then
                       return ret
                    end

                    local retAlgo = algType[findAPI_idx][0]
                    if find.verbose then
                       print(string.format(
                                "\n" .. API .. ": %d (ws limit: %d) mode = %s",
                                tonumber(retAlgo),
                                algWorkspaceLimit,
                                algSearchMode))
                    end
                    local bufSizeptr = ffi.new("size_t[1]")
                    ret = cudnn.call(getWSAlgos[findAPI_idx],
                                     cudnn.getHandle(),
                                     params[1], params[3], layer.convDesc[0], params[6],
                                     retAlgo, bufSizeptr)
                    local bufSize = tonumber(bufSizeptr[0])                 
  
                    if ret ~= 0 then
                       return ret
                    end
                    if find.verbose then
                       print(string.format(
                                "\n" .. getWSAlgos[findAPI_idx]  .. ": bufSize: %d, current ws: %d",
                                bufSize, tonumber(curWorkspaceSize)))
                    end
                    perfResults[0].algo = retAlgo
                    perfResults[0].memory = bufSize
                    perfResults[0].status = ret
                 end
              end

              if find.verbose then
                 print("\ncallCudnn: ", API, "returned ",  numPerfResults[0], " results , status = " , ret, "status[0] = " , perfResults[0].status, "\n")
              end

              if ret ~= 0 then
                 return ret
              end

              for r=0,numPerfResults[0]-1 do
                 local res = perfResults[r]
                 if res.status == 0 then
                    validResults = validResults+1
                    cachedAlgo[validResults] = { algo = tonumber(res.algo),
                                                 memory = tonumber(res.memory),
                                                 time = tonumber(res.time),
                                                 status = tonumber(res.status),
                                                 fallback = useFallback}
                    if find.verbose then
                       local fallback = ''
                       if (useFallback) then fallback = "[FALLBACK]"  end
                       print(string.format(
                                "\n" .. API .. " algo[%d]: %s (%d, status: %d), time: %.04f, memory: %8d, count: %d"
                                   .. " %s " .. cacheHit .. fallback,
                                validResults,
                                algoNames[findAPI_idx][cachedAlgo[validResults].algo+1], cachedAlgo[validResults].algo,  cachedAlgo[validResults].status,
                                cachedAlgo[validResults].time, cachedAlgo[validResults].memory, r, convDataString(layer)))
                    end
                 end
              end
              if validResults < 1 then
                 return 1
              end
              return 0
           end


           local function performanceFallback(layer)
              -- read conv descriptor
              local convDescData = layer.convDescData

              if convDescData and convDescData.dataType == "CUDNN_DATA_HALF" then
                 local savedResults = cachedAlgo
                 local savedNum = validResults
                 cachedAlgo = {}
                 validResults = 0
                 useFallback = true

                 -- update our record with fallback value
                 layer.convDescData.dataType = "CUDNN_DATA_FLOAT"
                 -- update the descriptor in CUDNN
                 cudnn.setConvolutionDescriptor(layer.convDescData, layer.convDesc)
                 -- do the actual call
                 local status = callCudnn(layer)
                 -- check if we got better results with float32
                 if status == 0 and validResults > 0 and cachedAlgo[1].time < savedResults[1].time then
                    if find.verbose or find.verboseFallback then
                       local msg = string.format("find.performanceFallback: found 32-bit float op is faster (%f) than FP16(%f), memory increase: %fM",
                                                 cachedAlgo[1].time, savedResults[1].time,
                                                 (tonumber(cachedAlgo[1].memory)-tonumber(savedResults[1].memory))/Meg)
                       fallbackWarning(layer, msg)
                    end
                    return
                 end
                 -- restore if we didn't
                cachedAlgo = savedResults
                validResults = savedNum
                -- update our record with fallback value
                layer.convDescData.dataType = "CUDNN_DATA_HALF"
                -- update the descriptor in CUDNN
                cudnn.setConvolutionDescriptor(layer.convDescData, layer.convDesc)

              end
           end

           -- do the actual call
           local status = callCudnn(layer)

           if status ~= 0 or validResults < 1 then
              if self.fallback and self.fallback(layer) then
                 useFallback = true
                 status = callCudnn(layer)
              end
              -- check again
              if status ~= 0  or validResults < 1 then
                 error (API .. ' failed, sizes: ' .. convDataString(layer))
              end
           else
              -- if we are running Find or FindEx in native fp16, check if this algo is actiually faster in pseudo
              if self.algoFamily ~= GetFamily then
                 performanceFallback(layer)
              end
           end
           self:store(layer, findAPI_idx, cachedAlgo)
           -- restore WS size if we fiddled with it
           if self.algoFamily ~= GetFamily then
              cudnn.setSharedWorkspaceSize(curWorkspaceSize)
           end
        end
        -- this may return different algo if size does not fit
        retAlgo = self:pickAlgoAndCalculateWorkspaceSize(cachedAlgo)
        if retAlgo > 0 then
           self:setCalculatedWorkspaceSize(true)
        else
           -- TODO: fallback to recalculate
           error("No algorithms found that would fit in free GPU memory")
           return -1
        end

        if cudnn.verbose or find.verbose then
           local freeMemory, totalMemory = cutorch.getMemoryUsage(self.id)
           local fallback = ""
           if (useFallback) then fallback = "[FALLBACK]"  end
           print(string.format(
                    "\n" .. API  .. ": %s(%d)[%d of %d] Workspace: %8fM (current ws size %fM, max: %dM free: %dM) %s" .. cacheHit .. fallback,
                    algoNames[findAPI_idx][cachedAlgo[retAlgo].algo+1], cachedAlgo[retAlgo].algo, retAlgo, #cachedAlgo,
                    tonumber(cachedAlgo[retAlgo].memory)/Meg, curWorkspaceSize/Meg, self.maxWorkspaceSize/Meg, freeMemory/Meg, convDataString(layer)))
        end
        return cachedAlgo[retAlgo].algo
end

function find:prepare(layer, input_slice, output_slice)
   local function shape(x)
      return table.concat(x:size():totable(),',')
   end
   local function vals(x)
      return table.concat(x,',')
   end
   layer.autotunerHash =
      '-dimA' .. shape(input_slice)
      ..' -filtA' .. shape(layer.weight)
      ..' '       .. shape(output_slice)
      ..' -padA'   .. vals(layer.pad)
      ..' -convStrideA' .. vals(layer.stride)
      .. ' ' .. cudnn.configmap(torch.type(layer.weight))

   layer.iteration = nil
   layer.input_slice = input_slice
   layer.output_slice = output_slice
end

local function setupWS(layer, params, algo, fn)
     local bufSizeptr = ffi.new("size_t[1]")
     cudnn.errcheck(getWSAlgos[fn],
                                     cudnn.getHandle(),
                                     params[1], params[3], layer.convDesc[0], params[6],
                                     algo, bufSizeptr)
     local bufSize = tonumber(bufSizeptr[0])                 
     cudnn.setSharedWorkspaceSize(bufSize, true)
end


function find:forwardAlgorithm(layer, params)
   if layer.fmode then
     setupWS(layer, params, layer.fmode, Fwd)
     return layer.fmode
   end
   local algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
   if layer.fastest_mode or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
   end
   return self:setupAlgo(layer, Fwd, algSearchMode, params)
end

function find:backwardFilterAlgorithm(layer, params)
   -- Check if we are in "sticky" mode
   if layer.bwmode then
     setupWS(layer, params, layer.bwmode, BwdFilter)
     return layer.bwmode
   end
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
   if layer.fastest_mode or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
   end
   local ret = self:setupAlgo(layer, BwdFilter, algSearchMode, params)
   return ret
end

function find:backwardDataAlgorithm(layer, params)
   -- Check if we are in "sticky" mode
   if layer.bdmode then
     setupWS(layer, params, layer.bdmode, BwdData)
     return layer.bdmode
   end
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
   if layer.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
   end
   return self:setupAlgo(layer, BwdData, algSearchMode, params)
end

find.reset()
return find
