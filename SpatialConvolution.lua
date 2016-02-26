local SpatialConvolution, parent =
    torch.class('cudnn.SpatialConvolution', 'nn.SpatialConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local autotunerCache = {}
autotunerCache[1] = {} -- forward
autotunerCache[2] = {} -- backwardFilter
autotunerCache[3] = {} -- backwardData

function SpatialConvolution:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH, groups)
    local delayedReset = self.reset
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
    self.reset = delayedReset
    self.padW = padW or 0
    self.padH = padH or 0
    self.groups = groups or 1
    assert(nInputPlane % self.groups == 0,
           'nInputPlane should be divisible by nGroups')
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.weight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane/self.groups, kH, kW)
    self:reset()
    -- should nil for serialization, the reset will still work
    self.reset = nil
end

-- if you change the configuration of the module manually, call this
function SpatialConvolution:resetWeightDescriptors()
    assert(torch.typename(self.weight) == 'torch.CudaTensor',
           'Only Cuda supported duh!')
    assert(torch.typename(self.bias) == 'torch.CudaTensor' or not self.bias,
           'Only Cuda supported duh!')
    -- for compatibility
    self.groups = self.groups or 1
    -- create filterDescriptor for weight
    self.weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
    errcheck('cudnnCreateFilterDescriptor', self.weightDesc)
    local desc = torch.IntTensor({self.nOutputPlane/self.groups,
                              self.nInputPlane/self.groups,
                              self.kH, self.kW})
    errcheck('cudnnSetFilterNdDescriptor', self.weightDesc[0],
             'CUDNN_DATA_FLOAT', 4,
             desc:data());
    local function destroyWDesc(d)
        errcheck('cudnnDestroyFilterDescriptor', d[0]);
    end
    ffi.gc(self.weightDesc, destroyWDesc)

    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end
end

function SpatialConvolution:fastest(mode)
    if mode == nil then mode = true end
    self.fastest_mode = mode
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function SpatialConvolution:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function SpatialConvolution:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function SpatialConvolution:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function SpatialConvolution:createIODescriptors(input)
    local batch = true
    if input:dim() == 3 then
        input = input:view(1, input:size(1), input:size(2), input:size(3))
        batch = false
    end
    assert(input:dim() == 4 and input:isContiguous());
    self.iSize = self.iSize or torch.LongStorage(4):fill(0)
    if not self.iDesc or not self.oDesc or
        input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
        self.iSize = input:size()

        -- resize gradInput
        if self.gradInput then self.gradInput:resizeAs(input); end
        assert(self.nInputPlane == input:size(2), 'input has to contain: '
                   .. self.nInputPlane
                   .. ' feature maps, but received input of size: '
                   .. input:size(1) .. ' x ' .. input:size(2) ..
                   ' x ' .. input:size(3) .. ' x ' .. input:size(4))

        -- create input descriptor
        local input_slice = {{},{1,self.nInputPlane/self.groups},{},{}}
        self.iDesc = cudnn.toDescriptor(input[input_slice])

        -- create conv descriptor
        self.convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
        errcheck('cudnnCreateConvolutionDescriptor', self.convDesc)
        local pad = torch.IntTensor({self.padH, self.padW})
        local stride = torch.IntTensor({self.dH, self.dW})
        local upscale = torch.IntTensor({1,1})
        errcheck('cudnnSetConvolutionNdDescriptor_v3', self.convDesc[0],
                 2, pad:data(),
                 stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
                 'CUDNN_DATA_FLOAT');
        local function destroyConvDesc(d)
            errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
        end
        ffi.gc(self.convDesc, destroyConvDesc)

        -- get output shape, resize output
        local oSize = torch.IntTensor(4)
        local oSizeD = oSize:data()
        errcheck('cudnnGetConvolutionNdForwardOutputDim',
                 self.convDesc[0], self.iDesc[0],
                 self.weightDesc[0], 4, oSizeD)
        oSize[2] = oSize[2] * self.groups
        self.output:resize(oSize:long():storage())

        -- create descriptor for output
        local output_slice = {{},{1,self.nOutputPlane/self.groups},{},{}}
        self.oDesc = cudnn.toDescriptor(self.output[output_slice])
        self.oDescForBias = cudnn.toDescriptor(self.output)

        -----------------------------------------------------------------------
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
        local autotunerHash = shape(self.weight) .. ';'
            .. shape(input[input_slice]) .. ';'
            .. shape(self.output[output_slice])

        local maxBufSize = 0

        -- create forwardAlgorithm descriptors
        local algType = ffi.new("cudnnConvolutionFwdAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
        local algWorkspaceLimit = self.workspace_limit
            or (self.nInputPlane * self.kH * self.kW * 4) -- 4 = sizeof int/float.

        if self.fastest_mode or cudnn.fastest == true then
            algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
        end

        if cudnn.benchmark then -- the manual auto-tuner is run
            if autotunerCache[1][autotunerHash] then
                algType[0] = autotunerCache[1][autotunerHash]
                if cudnn.verbose then
                    print('Using cached benchmark for: ', autotunerHash)
                end
            else
                local perfResults = ffi.new("cudnnConvolutionFwdAlgoPerf_t[?]", 1)
                local intt = torch.IntTensor(1);
                errcheck('cudnnFindConvolutionForwardAlgorithm',
                         cudnn.getHandle(),
                         self.iDesc[0], self.weightDesc[0],
                         self.convDesc[0], self.oDesc[0],
                         1, intt:data(), perfResults)
                algType[0] = perfResults[0].algo
                autotunerCache[1][autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "Autotuning        Forward: Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " Weight: %15s Input: %15s Output: %15s",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo),
                              shape(self.weight), shape(input[input_slice]),
                              shape(self.output[output_slice])))
                end
            end
        else
            errcheck('cudnnGetConvolutionForwardAlgorithm',
                     cudnn.getHandle(),
                     self.iDesc[0], self.weightDesc[0],
                     self.convDesc[0], self.oDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.fmode or algType[0]
        self.fwdAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionForwardWorkspaceSize',
                 cudnn.getHandle(),
                 self.iDesc[0], self.weightDesc[0],
                 self.convDesc[0], self.oDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        -- create backwardFilterAlgorithm descriptors
        local algType = ffi.new("cudnnConvolutionBwdFilterAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
        local algWorkspaceLimit = self.workspace_limit
            or (self.nInputPlane * self.kH * self.kW * 4) -- 4 = sizeof int/float.
        if self.fastest_mode  or cudnn.fastest == true then
            algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
        end

        if cudnn.benchmark then -- the manual auto-tuner is run
            if autotunerCache[2][autotunerHash] then
                algType[0] = autotunerCache[2][autotunerHash]
            else
                local perfResults = ffi.new("cudnnConvolutionBwdFilterAlgoPerf_t[?]", 1)
                local intt = torch.IntTensor(1);
                errcheck('cudnnFindConvolutionBackwardFilterAlgorithm',
                         cudnn.getHandle(),
                         self.iDesc[0], self.oDesc[0],
                         self.convDesc[0], self.weightDesc[0],
                         1, intt:data(), perfResults)
                algType[0] = perfResults[0].algo
                autotunerCache[2][autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "Autotuning backwardFilter: Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " Weight: %15s Input: %15s Output: %15s",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo),
                              shape(self.weight), shape(input[input_slice]),
                              shape(self.output[output_slice])))
                end
            end
        else
            errcheck('cudnnGetConvolutionBackwardFilterAlgorithm',
                     cudnn.getHandle(),
                     self.iDesc[0], self.oDesc[0],
                     self.convDesc[0], self.weightDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.bwmode or algType[0]
        self.bwdFilterAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionBackwardFilterWorkspaceSize',
                 cudnn.getHandle(),
                 self.iDesc[0], self.oDesc[0],
                 self.convDesc[0], self.weightDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        -- create backwardDataAlgorithm descriptors
        local algType = ffi.new("cudnnConvolutionBwdDataAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
        local algWorkspaceLimit = self.workspace_limit
            or (self.nInputPlane * self.kH * self.kW * 4) -- 4 = sizeof int/float.
        if self.fastest_mode or cudnn.fastest == true then
            algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
        end
        if cudnn.benchmark then -- the manual auto-tuner is run
            if autotunerCache[3][autotunerHash] then
                algType[0] = autotunerCache[3][autotunerHash]
            else
                local perfResults = ffi.new("cudnnConvolutionBwdDataAlgoPerf_t[?]", 1)
                local intt = torch.IntTensor(1);
                errcheck('cudnnFindConvolutionBackwardDataAlgorithm',
                         cudnn.getHandle(),
                         self.weightDesc[0], self.oDesc[0],
                         self.convDesc[0], self.iDesc[0],
                         1, intt:data(), perfResults)
                algType[0] = perfResults[0].algo
                autotunerCache[3][autotunerHash] = perfResults[0].algo
                if cudnn.verbose then
                    print(string.format(
                              "Autotuning   backwardData: Time: %3.5f Memory: %8d Algorithm: %d"
                                  .. " Weight: %15s Input: %15s Output: %15s\n",
                              perfResults[0].time, tonumber(perfResults[0].memory),
                              tonumber(perfResults[0].algo),
                              shape(self.weight), shape(input[input_slice]),
                              shape(self.output[output_slice])))
                end
            end
        else
            errcheck('cudnnGetConvolutionBackwardDataAlgorithm',
                     cudnn.getHandle(),
                     self.weightDesc[0], self.oDesc[0],
                     self.convDesc[0], self.iDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.bdmode or algType[0]
        self.bwdDataAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionBackwardDataWorkspaceSize',
                 cudnn.getHandle(),
                 self.weightDesc[0], self.oDesc[0],
                 self.convDesc[0], self.iDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        self.extraBuffer = self.extraBuffer or cudnn.getSharedWorkspace()
        self.extraBufferSizeInBytes = self.extraBuffer:nElement() * 4 -- float
        if maxBufSize > self.extraBufferSizeInBytes then
            self.extraBuffer:resize(math.ceil(maxBufSize/4))
            self.extraBufferSizeInBytes = maxBufSize
        end

        -----------------------------------------------------------------------
        -- create offsets for groups
        local iH, iW = input:size(3), input:size(4)
        local kH, kW = self.kH, self.kW
        local oH, oW = oSize[3], oSize[4]
        self.input_offset = self.nInputPlane / self.groups * iH * iW
        self.output_offset = self.nOutputPlane / self.groups * oH * oW
        self.weight_offset = self.nInputPlane / self.groups * self.nOutputPlane / self.groups * kH * kW

        if not batch then
            self.gradInput = self.gradInput:view(self.gradInput:size(2),
                                                 self.gradInput:size(3),
                                                 self.gradInput:size(4))
            self.output = self.output:view(self.output:size(2),
                                           self.output:size(3),
                                           self.output:size(4))
        end
    end
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function SpatialConvolution:updateOutput(input)
    if not self.weightDesc then self:resetWeightDescriptors() end
    input = makeContiguous(self, input)
    self:createIODescriptors(input)

    for g = 0, self.groups - 1 do
        errcheck('cudnnConvolutionForward', cudnn.getHandle(),
                 one:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.convDesc[0], self.fwdAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 zero:data(),
                 self.oDesc[0], self.output:data() + g*self.output_offset);
    end

    -- add bias
    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 one:data(), self.biasDesc[0], self.bias:data(),
                 one:data(), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end

    input, gradOutput = makeContiguous(self, input, gradOutput)
    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)

    for g = 0,self.groups - 1 do
        errcheck('cudnnConvolutionBackwardData_v3', cudnn.getHandle(),
                 one:data(),
                 self.weightDesc[0], self.weight:data() + g*self.weight_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdDataAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 zero:data(),
                 self.iDesc[0], self.gradInput:data() + g*self.input_offset);
    end
    return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or torch.FloatTensor(1):fill(1.0)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale

    input, gradOutput = makeContiguous(self, input, gradOutput)

    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)

    -- gradBias
    if self.bias then
        errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 one:data(),
                 self.biasDesc[0], self.gradBias:data())
    end

    for g = 0, self.groups - 1 do
        -- gradWeight
        errcheck('cudnnConvolutionBackwardFilter_v3', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.iDesc[0], input:data() + g*self.input_offset,
                 self.oDesc[0], gradOutput:data() + g*self.output_offset,
                 self.convDesc[0],
                 self.bwdFilterAlgType[0],
                 self.extraBuffer:data(), self.extraBufferSizeInBytes,
                 one:data(),
                 self.weightDesc[0], self.gradWeight:data() + g*self.weight_offset);
    end
end

function SpatialConvolution:clearDesc()
    self.weightDesc = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.algType = nil
    self.fwdAlgType = nil
    self.bwdDataAlgType = nil
    self.bwdFilterAlgType = nil
    self.extraBuffer = nil
    self.extraBufferSizeInBytes = nil
    self.scaleT = nil
end

function SpatialConvolution:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function SpatialConvolution:clearState()
   self:clearDesc()
   return nn.Module.clearState(self)
end
