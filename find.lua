local ffi = require 'ffi'

find = {}
find.__index = find

local function call(self, f, ...)
   local status = cudnn.call(f, ...)
   if cudnn.verbose and status ~= ffi.C.CUDNN_STATUS_SUCCESS then
      print(f .. " failed for sizes: " .. self.autotunerHash)
   end
   return status
end
find.call = call

local function errcheck(self, f, ...)
   local status = call(self, f, ...)
   if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
      local str = ffi.string(cudnn.C.cudnnGetErrorString(status))
      error('Error in CuDNN: ' .. str .. ' ('..f..')')
   end
end
find.errcheck = errcheck

-- FindEx State Machine and Cache (per device)
function find.create(id)
   local finder = {}
   setmetatable(finder,find)
   finder.id = id
   finder:resetStateMachine()
   finder:resetAlgorithmCache()
   return finder
end

-- FindEx State Machine cycle works as follows:
-- iteration #0(useDefaultWorkspaceSize) : call FindEx with default WS size (let everybody allocate I/O, weights etc)
-- iteration #1(useMaxWorkspaceSize) : call FindEx with maximum WS size, calculate common target WS using largest WS requested
-- iteration #2+(useCalculatedWorkspaceSize) : set calculated WS. call FindEx again with calculated WS size, cache the result
-- note: calculatedWorkspaceSize array is attribute of the cache (maximum WS of the cached algos) and reset separately

-- This resets SM of particular device to cycle 0 : useDefaultWorkspaceSize
function find:resetStateMachine()
   self.useDefaultWorkspaceSize = true
   self.useMaxWorkspaceSize = false
   self.useCalculatedWorkspaceSize = false
   self.iteration = 0
end

local finders = nil
-- this resets algorithm cache for device
function find:resetAlgorithmCache()
   self.calculatedWorkspaceSize  = {}
   self.maxWorkspaceSize = 0
   self.useFindEx = cudnn.useFindEx and (cudnn.benchmark or cudnn.fastest)
   self.autotunerCache = {{}, {}, {}}
end

function find.reset()
   cutorch.synchronize()
   finders = {}
end

function find.get()
   local device = cutorch.getDevice()
   local it = finders[device]
   if not it then
      it = find.create(device)
      finders[device] = it
   end
   return it
end

function find:lookup(layer, findAPI_idx)
   if self.useCalculatedWorkspaceSize or not self.useFindEx then
      return  self.autotunerCache[findAPI_idx][layer.autotunerHash]
   else
      return nil
   end
end

-- record algo, memory in cache
function find:store(layer, findAPI_idx, cachedAlgo)
--   print ("Storing ", findAPI_idx, layer.autotunerHash, cachedAlgo)
   self.autotunerCache[findAPI_idx][layer.autotunerHash] = cachedAlgo
end

function find:setMaxWorkspaceSize(reserve, fraction)
   if not reserve or reserve < cudnn.reservedGPUBytes then reserve = cudnn.reservedGPUBytes end
   local max_fraction =  cudnn.maxWorkspaceGPUMemPercent/100
   if not fraction or fraction > max_fraction then fraction = max_fraction end
   -- check current usage
   local freeMemory, totalMemory = cutorch.getMemoryUsage(self.id)

   local ws, curSize = cudnn.getSharedWorkspace()
   local newSize= (freeMemory+curSize-reserve) * fraction
   if (newSize > curSize) then
      self.maxWorkspaceSize = newSize
      cudnn.setSharedWorkspaceSize(newSize)
   else
      self.maxWorkspaceSize = curSize
   end
   self.useMaxWorkspaceSize = true
   if cudnn.verbose then
      print("setMaxWorkspaceSize Memory: ", freeMemory, totalMemory, self.maxWorkspaceSize)
   end
end

function find:setCalculatedWorkspaceSize(greater)
   for i,bytes in pairs (self.calculatedWorkspaceSize) do
      cudnn.setSharedWorkspaceSize(bytes, greater)
   end
   self.useCalculatedWorkspaceSize = true
end

-- adjusts workspace immediately in no FindEx
function find:registerWorkspaceSize(cachedAlgo)
    local stream = cutorch.getStream()
    if self.useFindEx then
       if not self.calculatedWorkspaceSize[stream] then
          self.calculatedWorkspaceSize[stream] = 0
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
                self.calculatedWorkspaceSize[stream] = algoSize + delta
                   if cudnn.verbose then
                      print("find:registerWorkspaceSize: calculated ", self.calculatedWorkspaceSize[stream], " delta = ", delta, "max : " , self.maxWorkspaceSize)
                   end
                return cachedAlgo[a].algo
             end
          else
                return cachedAlgo[a].algo
          end  -- delta
       end
       errcheck(false)
    else
       -- no FindEx - do not rely on find stored data
       cudnn.setSharedWorkspaceSize(cachedAlgo[1].memory, true)
       return cachedAlgo[1].algo
    end
end

function find:reserveBytes(layer)
   local reserve = cudnn.reservedGPUBytes
   -- todo: implement layer method returning memory allocation size
   reserve = reserve + 2*layer.weight:nElement()*layer.weight:elementSize()
   return reserve
end


function find:verifyWorkspaceSize(layer)
   local freeMemory, totalMemory = cutorch.getMemoryUsage(self.id)
   local reserve =  self:reserveBytes(layer)
   if freeMemory < reserve then
      -- let's make sure we still have space to reallocate our data
      cudnn.adjustSharedWorkspaceSize(freeMemory - reserve)
   end
end

function find:newIteration(layer)
   if self.useCalculatedWorkspaceSize or not self.useFindEx then
      --- end state
      return false
   end
   if not layer.iteration then layer.iteration = {0,0,0} end
   -- find last iteration
   local max_iter = 0
   for k,v in pairs(layer.iteration) do
      if v > max_iter then max_iter = v end
   end

   if (self.iteration < max_iter and max_iter > 1) then
      self.iteration = max_iter
      if cudnn.verbose then  print ("CUDNN Find SM: iteration ", self.iteration) end
      return true
   else
     return false
  end
end

function find:advanceStateMachine(layer, findAPI_idx)
   if not self.useFindEx then return end
   if self:newIteration(layer) then
      -- iteration changed, advance state machine
      if self.useMaxWorkspaceSize then
         if cudnn.verbose then  print ("SM: max->calculated ", self.calculatedWorkspaceSize) end
         self:setCalculatedWorkspaceSize()
         self.useMaxWorkspaceSize = false
      end
      if self.useDefaultWorkspaceSize then
         if self.useFindEx then
            if cudnn.verbose then print ("Find SM: default->max") end
            self:setMaxWorkspaceSize(self:reserveBytes(layer))
         else
            if cudnn.verbose then print ("Find SM: default->calculated ", self.calculatedWorkspaceSize) end
            self:setCalculatedWorkspaceSize(true)
         end
         self.useDefaultWorkspaceSize = false
      end
   end
   layer.iteration[findAPI_idx] = layer.iteration[findAPI_idx] + 1
end


local findNoExAlgos = {'cudnnFindConvolutionForwardAlgorithm', 'cudnnFindConvolutionBackwardFilterAlgorithm', 'cudnnFindConvolutionBackwardDataAlgorithm'}
local findExAlgos = {'cudnnFindConvolutionForwardAlgorithmEx', 'cudnnFindConvolutionBackwardFilterAlgorithmEx', 'cudnnFindConvolutionBackwardDataAlgorithmEx'}

function find:setupAlgo(layer, algo_t, perf_t, findAPI_idx, getAPI, wsAPI, algSearchMode, params)
        local findAPI = findNoExAlgos[findAPI_idx]
        local findExAPI = findExAlgos[findAPI_idx]
        local retAlgo
        local cacheHit = '[found in cache]'

        -- advance state machine
        self:advanceStateMachine(layer, findAPI_idx)

        local curFindAPI = findAPI
        if self.useFindEx then
           curFindAPI = findExAPI
        end

        local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()
        local cachedAlgo =  self:lookup(layer, findAPI_idx)

        if not cachedAlgo then
           cacheHit = ''
           cachedAlgo = {}

           local perfResults = ffi.new(perf_t, 10)

           if cudnn.benchmark or cudnn.fastest then -- the manual auto-tuner is run
              local intt = torch.IntTensor(1)
              if cudnn.useFindEx then
                 -- use clone for weights when looking for backward filter algo
                 if findAPI_idx == 2 then
                    params[7] = params[7]:clone()
                 end
                 errcheck(layer, findExAPI,
                          cudnn.getHandle(),
                          params[1], params[2]:data(), params[3], params[4]:data(), params[5], params[6], params[7]:data(),
                          10, intt:data(), perfResults, extraBuffer, extraBufferSize)
              else
                 curFindAPI = findAPI
                 errcheck(layer, findAPI,
                          cudnn.getHandle(),
                          params[1], params[3], params[5], params[6],
                          10, intt:data(), perfResults)
              end

              local validResults = 0
              for r=0,intt[1]-1 do
                 local res = perfResults[r]
                 if res.status == 0 then
                    validResults = validResults+1
                    cachedAlgo[validResults] = { algo = tonumber(res.algo),
                                                 memory = tonumber(res.memory),
                                                 time = tonumber(res.time),
                                                 status = tonumber(res.status) }
                    --- todo: use fallback if status ~= 0
                    if cudnn.verbose and find.verbose then
                       print(string.format(
                                "\n" .. curFindAPI .. " algo: %d (status: %d), memory: %8d, count: %d"
                                   .. " hash: %45s " .. cacheHit,
                                cachedAlgo[validResults].algo,  cachedAlgo[validResults].status,
                                cachedAlgo[validResults].memory, r, layer.autotunerHash))
                    end
                 end
              end

           else
              curFindAPI=getAPI
              local algWorkspaceLimit = layer.workspace_limit
                 or (layer.nInputPlane * layer.kH * layer.kW * layer.weight.elementSize())

              local algType = ffi.new(algo_t, 1)
              errcheck(layer, getAPI,
                       cudnn.getHandle(),
                       params[1], params[3], params[5], params[6],
                       algSearchMode, algWorkspaceLimit, algType)

              if cudnn.verbose then
                 print(string.format(
                          "\n" .. getAPI .. ": %d (ws limit: %d) mode = %s",
                          tonumber(algType[0]),
                          algWorkspaceLimit,
                          algSearchMode))
              end


              local bufSize = torch.LongTensor(1)
              errcheck(layer, wsAPI,
                       cudnn.getHandle(),
                       params[1], params[3], params[5], params[6],
                       algType[0], bufSize:data())
              if cudnn.verbose then
                 print(string.format(
                          "\n" .. wsAPI  .. ": bufSize: %d, current ws: %d",
                          tonumber(bufSize[1]), tonumber(extraBufferSize)))
              end
              cachedAlgo[1] = {algo = tonumber(algType[0]),
                               memory = tonumber(bufSize[1]),
                               status = 0}
           end
           self:store(layer, findAPI_idx, cachedAlgo)
        end
        -- this may return different algo if size does not fit
        retAlgo = self:registerWorkspaceSize(cachedAlgo)

        if cudnn.verbose then
           local freeMemory, totalMemory = cutorch.getMemoryUsage(self.id)
           print(string.format(
                    "\n" .. curFindAPI  .. ": %d(%d) Workspace: %8d (current ws size %d, free: %d)  hash: %45s" .. cacheHit,
                    retAlgo, tonumber(cachedAlgo[1].algo), tonumber(cachedAlgo[1].memory), extraBufferSize, freeMemory, layer.autotunerHash))
        end
        return retAlgo
end

function find:prepare(layer, input_slice, output_slice)
   local function shape(x)
      return table.concat(x:size():totable(),'x')
   end
   layer.autotunerHash = shape(layer.weight) .. ';'
      .. shape(input_slice) .. ';'
      .. shape(output_slice) .. "[" .. layer.padH .. ":" .. layer.padW .. ']' .. cudnn.configmap(torch.type(layer.weight))

   layer:resetMode()
   layer.iteration = nil
   layer.input_slice = input_slice
   layer.output_slice = output_slice
end

function find:forwardAlgorithm(layer, params)
   local algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
   if layer.fastest_mode  or cudnn.benchmark == true or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
   end
   -- supply a temporary for findEx
   return self:setupAlgo(layer,"cudnnConvolutionFwdAlgo_t[?]", "cudnnConvolutionFwdAlgoPerf_t[?]",
                         1, 'cudnnGetConvolutionForwardAlgorithm',
                         'cudnnGetConvolutionForwardWorkspaceSize', algSearchMode, params)
end

function find:backwardFilterAlgorithm(layer, params)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
   if layer.fastest_mode or cudnn.benchmark == true or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
   end
   local ret = self:setupAlgo(layer,"cudnnConvolutionBwdFilterAlgo_t[?]", "cudnnConvolutionBwdFilterAlgoPerf_t[?]",
                              2, 'cudnnGetConvolutionBackwardFilterAlgorithm',
                              'cudnnGetConvolutionBackwardFilterWorkspaceSize', algSearchMode, params)
   return ret
end

function find:backwardDataAlgorithm(layer, params)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
   if layer.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
   end
   return self:setupAlgo(layer,"cudnnConvolutionBwdDataAlgo_t[?]", "cudnnConvolutionBwdDataAlgoPerf_t[?]",
                         3, 'cudnnGetConvolutionBackwardDataAlgorithm',
                         'cudnnGetConvolutionBackwardDataWorkspaceSize', algSearchMode, params)

end

find.reset()
return find
