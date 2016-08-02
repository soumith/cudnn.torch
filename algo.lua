local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local algo = {}
local autotunerCache = {}
autotunerCache[1] = {} -- forward
autotunerCache[2] = {} -- backwardFilter
autotunerCache[3] = {} -- backwardData

local function setupAlgo(self, algo_t, perf_t, findAPI, getAPI, wsAPI, algSearchMode, params)
        -- create forwardAlgorithm descriptors
        local algType = ffi.new(algo_t, 1)

        self.extraBuffer = self.extraBuffer or cudnn.getSharedWorkspace()
        self.extraBufferSizeInBytes = self.extraBuffer:nElement() * self.extraBuffer.elementSize()

        local algWorkspaceLimit = self.workspace_limit
           or (self.nInputPlane * self.kH * self.kW * self.extraBuffer.elementSize())


        if cudnn.benchmark then -- the manual auto-tuner is run
            if autotunerCache[1][self.autotunerHash] then
                algType[0] = autotunerCache[1][self.autotunerHash]
                if cudnn.verbose then
                   print('\nAutotuning ', algo_t, ' using cached algo = ' , algType[0] , ' for: ', self.autotunerHash)
                end
            else
                local perfResults = ffi.new(perf_t, 1)
                local intt = torch.IntTensor(1)
                errcheck(findAPI,
                         cudnn.getHandle(),
                         params[1], params[2], params[3], params[4],
                         1, intt:data(), perfResults)
                algType[0] = perfResults[0].algo
                autotunerCache[1][self.autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "\nAutotuning " .. algo_t .. " Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " hash: %45s\n",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo), self.autotunerHash ))

                end
            end
        else
            errcheck(getAPI,
                     cudnn.getHandle(),
                     params[1], params[2], params[3], params[4],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        local bufSize = torch.LongTensor(1)
        errcheck(wsAPI,
                 cudnn.getHandle(),
                 params[1], params[2], params[3], params[4],
                 algType[0], bufSize:data())
        if self.extraBufferSizeInBytes < bufSize[1] then
           self.extraBuffer:resize(math.ceil(bufSize[1]/self.extraBuffer.elementSize()))
           self.extraBufferSizeInBytes = bufSize[1]
        end
        return algType
end

function algo.setupForwardAlgorithm(self, input_slice, output_slice)
   local function shape(x)
      local sz = x:size()
      local str = ''
      for i=1,sz:size() do
         str = str .. sz[i] .. 'x'
      end
      if #str > 0 then
         str = str:sub(1, #str-1)
      end
      return str
   end

   self.autotunerHash = shape(self.weight) .. ';'
      .. shape(input_slice) .. ';'
      .. shape(output_slice)

   local algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
   if self.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
   end

   local params = { self.iDesc[0], self.weightDesc[0], self.convDesc[0], self.oDesc[0] }
   local algType = setupAlgo(self,"cudnnConvolutionFwdAlgo_t[?]", "cudnnConvolutionFwdAlgoPerf_t[?]",
                             'cudnnFindConvolutionForwardAlgorithm', 'cudnnGetConvolutionForwardAlgorithm',
                             'cudnnGetConvolutionForwardWorkspaceSize', algSearchMode, params)
   algType[0] = self.fmode or algType[0]
   self.fwdAlgType = algType
   self.bwdDataAlgType = nil
   self.bwdFilterAlgType = nil
end

function algo.setupBackwardFilterAlgorithm(self)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
   if self.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
   end
   local params = { self.iDesc[0], self.oDesc[0], self.convDesc[0], self.weightDesc[0] }
   local algType = setupAlgo(self,"cudnnConvolutionBwdFilterAlgo_t[?]", "cudnnConvolutionBwdFilterAlgoPerf_t[?]",
                                     'cudnnFindConvolutionBackwardFilterAlgorithm', 'cudnnGetConvolutionBackwardFilterAlgorithm',
                                     'cudnnGetConvolutionBackwardFilterWorkspaceSize', algSearchMode,
                                     params)
   algType[0] = self.bwmode or algType[0]
   self.bwdFilterAlgType = algType
end

function algo.setupBackwardDataAlgorithm(self)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
   if self.fastest_mode  or cudnn.fastest == true then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
   end
   local params =  { self.weightDesc[0], self.oDesc[0], self.convDesc[0], self.iDesc[0] }
   local algType = setupAlgo(self,"cudnnConvolutionBwdDataAlgo_t[?]", "cudnnConvolutionBwdDataAlgoPerf_t[?]",
                             'cudnnFindConvolutionBackwardDataAlgorithm', 'cudnnGetConvolutionBackwardDataAlgorithm',
                             'cudnnGetConvolutionBackwardDataWorkspaceSize', algSearchMode, params)
   algType[0] = self.bdmode or algType[0]
   self.bwdDataAlgType = algType
end

return algo
