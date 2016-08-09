local ffi = require 'ffi'

local algo = {}

algo.findNoExAlgos = {'cudnnFindConvolutionForwardAlgorithm', 'cudnnFindConvolutionBackwardFilterAlgorithm', 'cudnnFindConvolutionBackwardDataAlgorithm'}
algo.findExAlgos = {'cudnnFindConvolutionForwardAlgorithmEx', 'cudnnFindConvolutionBackwardFilterAlgorithmEx', 'cudnnFindConvolutionBackwardDataAlgorithmEx'}

algo.autotunerCache = {{}, {}, {}}
algo.findAlgos = nil

local function call(self, f, ...)
   local status = cudnn.call(f, ...)
   if cudnn.verbose and status ~= ffi.C.CUDNN_STATUS_SUCCESS then
      print(f .. " failed for sizes: " .. self.autotunerHash)
   end
   return status
end
algo.call = call

local function errcheck(self, f, ...)
   local status = call(self, f, ...)
   if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
      local str = ffi.string(cudnn.C.cudnnGetErrorString(status))
      error('Error in CuDNN: ' .. str .. ' ('..f..')')
      return false
   end
   return true
end
algo.errcheck = errcheck

local function initCache(useEx)
   if useEx then
      algo.findAlgos = algo.findExAlgos
   else
      algo.findAlgos = algo.findNoExAlgos
   end
end

local function setupAlgo(self, algo_t, perf_t, findAPI_idx, getAPI, wsAPI, algSearchMode, params)
        -- recheck if cudnn.useFindEx was reset
        initCache(cudnn.useFindEx)
        local findAPI = algo.findAlgos[findAPI_idx]
        self.extraBuffer = self.extraBuffer or cudnn.getSharedWorkspace()
        local extraBufferSizeInBytes = self.extraBuffer:nElement() * self.extraBuffer.elementSize()
        local cachedAlgo = algo.autotunerCache[findAPI_idx][self.autotunerHash]
        local cacheHit = '[found in cache]'
        if not cachedAlgo then
           cacheHit = ''
           cachedAlgo = {}
           local perfResults = ffi.new(perf_t, 1)
           if cudnn.benchmark or cudnn.fastest then -- the manual auto-tuner is run
              local intt = torch.IntTensor(1)
              local status
              if cudnn.useFindEx then
                 status = algo.call(self, findAPI,
                                     cudnn.getHandle(),
                                     params[1], params[2], params[3], params[4], params[5], params[6], params[7],
                                     1, intt:data(), perfResults, self.extraBuffer:data(), self.extraBuffer:nElement() * self.extraBuffer.elementSize())

              else
                 status = algo.call(self, findAPI,
                          cudnn.getHandle(),
                          params[1], params[3], params[5], params[6],
                          1, intt:data(), perfResults)
              end

              if status == ffi.C.CUDNN_STATUS_SUCCESS then
                 cachedAlgo.algo = tonumber(perfResults[0].algo)
                 cachedAlgo.memory = tonumber(perfResults[0].memory)
                 cachedAlgo.time = tonumber(perfResults[0].time)
                 cachedAlgo.status = tonumber(perfResults[0].status)
              else
                 cachedAlgo.algo =0
                 cachedAlgo.memory = 0
                 cachedAlgo.time = 0
                 cachedAlgo.status = tonumber(status)
              end
              if cudnn.verbose then
                 print(string.format(
                          "\n" .. findAPI .. " Time: %3.5f Memory: %8d Algorithm: %d(out of %d, status: %d)"
                             .. " hash: %45s",
                          cachedAlgo.time, cachedAlgo.memory,
                          cachedAlgo.algo, intt[1], cachedAlgo.status, self.autotunerHash ))

              end
           else
              findAPI=getAPI
              local algType = ffi.new(algo_t, 1)
              local algWorkspaceLimit = self.workspace_limit
                 or (self.nInputPlane * self.kH * self.kW * self.weight.elementSize())
              local status = algo.call(self, getAPI,
                                       cudnn.getHandle(),
                                       params[1], params[3], params[5], params[6],
                                       algSearchMode, algWorkspaceLimit, algType)


              if cudnn.verbose then
                 print(string.format(
                          "\n" .. getAPI .. ": %d (ws limit: %d) mode = %s status=%d",
                          tonumber(algType[0]),
                          algWorkspaceLimit,
                          algSearchMode, tonumber(status)))
              end

              if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
                 cachedAlgo.algo =0
                 cachedAlgo.memory = 0
              else
                 cachedAlgo.algo = tonumber(algType[0])
                 local bufSize = torch.LongTensor(1)
                 status = algo.call(self, wsAPI,
                                    cudnn.getHandle(),
                                    params[1], params[3], params[5], params[6],
                                    algType[0], bufSize:data())
                 if cudnn.verbose then
                    print(string.format(
                             "\n" .. wsAPI  .. ": bufSize: %d, current ws: %d, status: %d",
                             tonumber(bufSize[1]), tonumber(extraBufferSizeInBytes), tonumber(status)))
                 end
                 if status ~= ffi.C.CUDNN_STATUS_SUCCESS then
                    cachedAlgo.memory = 0
                 else
                    cachedAlgo.memory = tonumber(bufSize[1])
                 end
              end
           end
           algo.autotunerCache[findAPI_idx][self.autotunerHash] = cachedAlgo
        end

        if extraBufferSizeInBytes < cachedAlgo.memory then
           self.extraBuffer:resize((tonumber(cachedAlgo.memory)+self.extraBuffer.elementSize())/self.extraBuffer.elementSize())
        end

        if cudnn.verbose then
           print(string.format(
                    "\n" .. findAPI  .. ": %d Workspace: %8d  hash: %45s" .. cacheHit,
                    tonumber(cachedAlgo.algo), tonumber(cachedAlgo.memory), self.autotunerHash))
        end
        return cachedAlgo.algo
end

function algo.prepareHash(self, input_slice, output_slice)
   local function shape(x)
      return table.concat(x:size():totable(),'x')
   end
   self.autotunerHash = shape(self.weight) .. ';'
      .. shape(input_slice) .. ';'
      .. shape(output_slice)

   self.fwdAlgType = nil
   self.bwdDataAlgType = nil
   self.bwdFilterAlgType = nil
   self.input_slice = input_slice
   self.output_slice = output_slice
end

function algo.setupForwardAlgorithm(self, params)
   local algSearchMode
   if self.fastest_mode  or cudnn.benchmark == true or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
   else
      algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
   end

   params = params or { self.iDesc[0], self.input_slice:data(), self.weightDesc[0], self.weight:data(), self.convDesc[0], self.oDesc[0], self.output_slice:data() }
   self.fwdAlgType = self.fmode or
      setupAlgo(self,"cudnnConvolutionFwdAlgo_t[?]", "cudnnConvolutionFwdAlgoPerf_t[?]",
                1, 'cudnnGetConvolutionForwardAlgorithm',
                'cudnnGetConvolutionForwardWorkspaceSize', algSearchMode, params)
end

function algo.setupBackwardFilterAlgorithm(self, params)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
   if self.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
   end
   params = params or { self.iDesc[0], self.input_slice:data(), self.oDesc[0], self.output_slice:data(), self.convDesc[0], self.weightDesc[0], self.weight:data() }
   self.bwdFilterAlgType = self.bwmode or
      setupAlgo(self,"cudnnConvolutionBwdFilterAlgo_t[?]", "cudnnConvolutionBwdFilterAlgoPerf_t[?]",
                2, 'cudnnGetConvolutionBackwardFilterAlgorithm',
                'cudnnGetConvolutionBackwardFilterWorkspaceSize', algSearchMode,
                params)
end

function algo.setupBackwardDataAlgorithm(self, params)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
   if self.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
   end
   params =  params or { self.weightDesc[0], self.weight:data(), self.oDesc[0], self.output_slice:data(), self.convDesc[0], self.iDesc[0], self.input_slice:data() }
   self.bwdDataAlgType = self.bdmode or
      setupAlgo(self,"cudnnConvolutionBwdDataAlgo_t[?]", "cudnnConvolutionBwdDataAlgoPerf_t[?]",
                3, 'cudnnGetConvolutionBackwardDataAlgorithm',
                'cudnnGetConvolutionBackwardDataWorkspaceSize', algSearchMode, params)
end

return algo
