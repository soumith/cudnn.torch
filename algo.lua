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
   end
end
algo.errcheck = errcheck

local function setupAlgo(self, algo_t, perf_t, findAPI_idx, getAPI, wsAPI, algSearchMode, params)
        local findAPI = algo.findNoExAlgos[findAPI_idx]
        local findExAPI = algo.findExAlgos[findAPI_idx]

        self.extraBuffer = self.extraBuffer or cudnn.getSharedWorkspace()
        local extraBufferSizeInBytes = self.extraBuffer:nElement() * self.extraBuffer.elementSize()
        local algWorkspaceLimit = self.workspace_limit
           or (self.nInputPlane * self.kH * self.kW * self.weight.elementSize())

        local cacheHit = '[found in cache]'
        local cachedAlgo = algo.autotunerCache[findAPI_idx][self.autotunerHash]
        if not cachedAlgo then
           cacheHit = ''
           cachedAlgo = {}
           local perfResults = ffi.new(perf_t, 1)

           if cudnn.benchmark or cudnn.fastest then -- the manual auto-tuner is run
              local intt = torch.IntTensor(1)
              errcheck(self, findAPI,
                       cudnn.getHandle(),
                       params[1], params[3], params[5], params[6],
                       1, intt:data(), perfResults)

              cachedAlgo.algo = tonumber(perfResults[0].algo)
              cachedAlgo.memory = tonumber(perfResults[0].memory)
              cachedAlgo.status = tonumber(perfResults[0].status)

              if cudnn.verbose then
                 print(string.format(
                          "\n" .. findAPI .. " algo: %d (status: %d), memory: %8d"
                             .. " hash: %45s " .. cacheHit,
                          cachedAlgo.algo,  cachedAlgo.status, cachedAlgo.memory, self.autotunerHash))

              end
           else
              findAPI=getAPI
              local algType = ffi.new(algo_t, 1)
              errcheck(self, getAPI,
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

              cachedAlgo.algo = tonumber(algType[0])
              local bufSize = torch.LongTensor(1)
              errcheck(self, wsAPI,
                       cudnn.getHandle(),
                       params[1], params[3], params[5], params[6],
                       algType[0], bufSize:data())
              if cudnn.verbose then
                 print(string.format(
                          "\n" .. wsAPI  .. ": bufSize: %d, current ws: %d",
                          tonumber(bufSize[1]), tonumber(extraBufferSizeInBytes)))
              end
              cachedAlgo.memory = tonumber(bufSize[1])
           end
           algo.autotunerCache[findAPI_idx][self.autotunerHash] = cachedAlgo
        end

        if cudnn.verbose then
           print(string.format(
                    "\n" .. findAPI  .. ": %d Workspace: %8d (current ws size %d)  hash: %45s" .. cacheHit,
                    tonumber(cachedAlgo.algo), tonumber(cachedAlgo.memory), extraBufferSizeInBytes, self.autotunerHash))
        end

        if extraBufferSizeInBytes < cachedAlgo.memory then
           extraBufferSizeInBytes = cachedAlgo.memory
           if self.workspace_limit and extraBufferSizeInBytes > self.workspace_limit then
              extraBufferSizeInBytes = self.workspace_limit
           end
        end

        if extraBufferSizeInBytes > self.extraBuffer:elementSize()*self.extraBuffer:nElement() then
           -- todo: how to check failure here ?
           self.extraBuffer:resize((extraBufferSizeInBytes+(self.extraBuffer:elementSize())-1)/self.extraBuffer:elementSize())
           local cant_resize=false
           if cudnn.useFindEx and cant_resize then
              errcheck(self, findExAPI,
                       cudnn.getHandle(),
                       params[1], params[2]:data(), params[3], params[4]:data(), params[5], params[6], params[7]:data(),
                       1, intt:data(), perfResults, self.extraBuffer:data(), self.extraBuffer:nElement() * self.extraBuffer.elementSize())
              cachedAlgo.algo = tonumber(perfResults[0].algo)
              cachedAlgo.memory = tonumber(perfResults[0].memory)
              cachedAlgo.status = tonumber(perfResults[0].status)
           end
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
   params = params or { self.iDesc[0], self.input_slice, self.weightDesc[0], self.weight, self.convDesc[0], self.oDesc[0], self.output_slice}
   -- supply a temporary for findEx
   if cudnn.useFindEx and (cudnn.benchmark or cudnn.fastest) and not self.fmode then
      params[7]=params[7]:clone()
   end
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
   params = params or { self.iDesc[0], self.input_slice, self.oDesc[0], self.output_slice, self.convDesc[0], self.weightDesc[0], self.weight}
   -- supply a temporary for findEx
   if cudnn.useFindEx and (cudnn.benchmark or cudnn.fastest) and not self.bdmode then
      params[7]=params[7]:clone()
   end
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
   params =  params or { self.weightDesc[0], self.weight, self.oDesc[0], self.output_slice, self.convDesc[0], self.iDesc[0], self.input_slice }
   -- supply a temporary for findEx
   if cudnn.useFindEx and (cudnn.benchmark or cudnn.fastest) and not self.bdmode then
      params[7]=params[7]:clone()
   end
   self.bwdDataAlgType = self.bdmode or
      setupAlgo(self,"cudnnConvolutionBwdDataAlgo_t[?]", "cudnnConvolutionBwdDataAlgoPerf_t[?]",
                3, 'cudnnGetConvolutionBackwardDataAlgorithm',
                'cudnnGetConvolutionBackwardDataWorkspaceSize', algSearchMode, params)
end

return algo
