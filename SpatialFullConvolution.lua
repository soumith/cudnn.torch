local SpatialFullConvolution, parent =
    torch.class('cudnn.SpatialFullConvolution', 'nn.SpatialFullConvolution')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local autotunerCache = {}
autotunerCache[1] = {} -- forward
autotunerCache[2] = {} -- backwardFilter
autotunerCache[3] = {} -- backwardData

function SpatialFullConvolution:__init(...)
    parent.__init(self, ...)
    self.iSize = torch.LongStorage(4):fill(0)
end

-- if you change the configuration of the module manually, call this
function SpatialFullConvolution:resetWeightDescriptors()
    assert(cudnn.typemap[torch.typename(self.weight)], 'Only Cuda supported duh!')
    assert(cudnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Cuda supported duh!')
    -- create filterDescriptor for weight
    self.weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
    errcheck('cudnnCreateFilterDescriptor', self.weightDesc)
    local desc = torch.IntTensor({self.nInputPlane,
                                  self.nOutputPlane,
                                  self.kH, self.kW})
    errcheck('cudnnSetFilterNdDescriptor', self.weightDesc[0],
             cudnn.typemap[torch.typename(self.weight)], 'CUDNN_TENSOR_NCHW', 4,
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

function SpatialFullConvolution:fastest(mode)
    if mode == nil then mode = true end
    self.fastest_mode = mode
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function SpatialFullConvolution:setMode(fmode, bdmode, bwmode)
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

function SpatialFullConvolution:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function SpatialFullConvolution:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function SpatialFullConvolution:createIODescriptors(input)
    local batch = true
    if input:dim() == 3 then
        input = input:view(1, input:size(1), input:size(2), input:size(3))
        batch = false
    end
    assert(input:dim() == 4 and input:isContiguous());
    if not self.iDesc or not self.oDesc or
        input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
        self.iSize:copy(input:size())

        assert(self.nInputPlane == input:size(2), 'input has to contain: '
                   .. self.nInputPlane
                   .. ' feature maps, but received input of size: '
                   .. input:size(1) .. ' x ' .. input:size(2) ..
                   ' x ' .. input:size(3) .. ' x ' .. input:size(4))

        -- create input descriptor
        local input_slice = {{},{1,self.nInputPlane},{},{}}
        self.iDesc = cudnn.toDescriptor(input[input_slice])

        -- create conv descriptor
        self.convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
        errcheck('cudnnCreateConvolutionDescriptor', self.convDesc)
        local pad = torch.IntTensor({self.padH, self.padW})
        local stride = torch.IntTensor({self.dH, self.dW})
        local upscale = torch.IntTensor({1,1})
        errcheck('cudnnSetConvolutionNdDescriptor', self.convDesc[0],
                 2, pad:data(),
                 stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
                 cudnn.configmap(torch.type(self.weight)));
        local function destroyConvDesc(d)
            errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
        end
        ffi.gc(self.convDesc, destroyConvDesc)

        -- get output shape, resize output
        local iwidth = input:size(4)
        local iheight = input:size(3)
        local owidth = (iwidth - 1) * self.dW - 2*self.padW + self.kW + self.adjW
        local oheight = (iheight - 1) * self.dH - 2*self.padH + self.kH + self.adjH
        local oSize = torch.IntTensor({input:size(1), self.nOutputPlane, oheight, owidth})
        self.output:resize(oSize:long():storage())

        -- create descriptor for output
        local output_slice = {{},{1,self.nOutputPlane},{},{}}
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
            or (self.nOutputPlane * self.kH * self.kW * 4) -- 4 = sizeof int/float.

        if self.fastest_mode or cudnn.fastest == true then
            algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
        end

        if cudnn.benchmark then -- the manual auto-tuner is run
            if autotunerCache[1][autotunerHash] then
                algType[0] = autotunerCache[1][autotunerHash]
                if cudnn.verbose then
                   print('Autotuning SFC: using cached algo = ', algType[0], ' for: ', autotunerHash)
                end
            else
                local perfResults = ffi.new("cudnnConvolutionFwdAlgoPerf_t[?]", 1)
                local intt = torch.IntTensor(1);
                errcheck('cudnnFindConvolutionForwardAlgorithm',
                         cudnn.getHandle(),
                         self.oDesc[0], self.weightDesc[0],
                         self.convDesc[0], self.iDesc[0],
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
                     self.oDesc[0], self.weightDesc[0],
                     self.convDesc[0], self.iDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.fmode or algType[0]
        self.fwdAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionForwardWorkspaceSize',
                 cudnn.getHandle(),
                 self.oDesc[0], self.weightDesc[0],
                 self.convDesc[0], self.iDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        -- create backwardFilterAlgorithm descriptors
        local algType = ffi.new("cudnnConvolutionBwdFilterAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
        local algWorkspaceLimit = self.workspace_limit
            or (self.nOutputPlane * self.kH * self.kW * 4) -- 4 = sizeof int/float.
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
                         self.oDesc[0], self.iDesc[0],
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
                     self.oDesc[0], self.iDesc[0],
                     self.convDesc[0], self.weightDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.bwmode or algType[0]
        self.bwdFilterAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionBackwardFilterWorkspaceSize',
                 cudnn.getHandle(),
                 self.oDesc[0], self.iDesc[0],
                 self.convDesc[0], self.weightDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        -- create backwardDataAlgorithm descriptors
        local algType = ffi.new("cudnnConvolutionBwdDataAlgo_t[?]", 1)
        local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
        local algWorkspaceLimit = self.workspace_limit
            or (self.nOutputPlane * self.kH * self.kW * 4) -- 4 = sizeof int/float.
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
                         self.weightDesc[0], self.iDesc[0],
                         self.convDesc[0], self.oDesc[0],
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
                     self.weightDesc[0], self.iDesc[0],
                     self.convDesc[0], self.oDesc[0],
                     algSearchMode, algWorkspaceLimit, algType)
        end
        algType[0] = self.bdmode or algType[0]
        self.bwdDataAlgType = algType
        local bufSize = torch.LongTensor(1)
        errcheck('cudnnGetConvolutionBackwardDataWorkspaceSize',
                 cudnn.getHandle(),
                 self.weightDesc[0], self.iDesc[0],
                 self.convDesc[0], self.oDesc[0],
                 algType[0], bufSize:data())
        maxBufSize = math.max(maxBufSize, bufSize[1])

        self.extraBuffer = self.extraBuffer or cudnn.getSharedWorkspace()
        self.extraBufferSizeInBytes = self.extraBuffer:nElement() * 4 -- float
        if maxBufSize > self.extraBufferSizeInBytes then
            self.extraBuffer:resize(math.ceil(maxBufSize/4))
            self.extraBufferSizeInBytes = maxBufSize
        end

        if not batch then
            self.gradInput:set(self.gradInput:view(self.gradInput:size(2),
                                                   self.gradInput:size(3),
                                                   self.gradInput:size(4)))
            self.output:set(self.output:view(self.output:size(2),
                                             self.output:size(3),
                                             self.output:size(4)))
        end
    end
end




function SpatialFullConvolution:updateOutput(input)
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)

    -- Because SpatialFullConvolution is performing the adjoint of the forward
    -- convolution operator, we need to swap the forward and backward passes.
    errcheck('cudnnConvolutionBackwardData', cudnn.getHandle(),
             cudnn.scalar(input, 1),
             self.weightDesc[0], self.weight:data(),
             self.iDesc[0], input:data(),
             self.convDesc[0], self.bwdDataAlgType[0],
             self.extraBuffer:data(), self.extraBufferSizeInBytes,
             cudnn.scalar(input, 0),
             self.oDesc[0], self.output:data())

    -- add bias
    if self.bias then
        errcheck('cudnnAddTensor', cudnn.getHandle(),
                 cudnn.scalar(input, 1), self.biasDesc[0], self.bias:data(),
                 cudnn.scalar(input, 1), self.oDescForBias[0], self.output:data())
    end

    return self.output
end

function SpatialFullConvolution:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.gradInput:resizeAs(input)

    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4, 'gradOutput has to be 3D or 4D');
    assert(gradOutput:isContiguous(), 'gradOutput has to be contiguous')
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)

    errcheck('cudnnConvolutionForward', cudnn.getHandle(),
             cudnn.scalar(input, 1),
             self.oDesc[0], gradOutput:data(),
             self.weightDesc[0], self.weight:data(),
             self.convDesc[0],
             self.fwdAlgType[0],
             self.extraBuffer:data(), self.extraBufferSizeInBytes,
             cudnn.scalar(input, 0),
             self.iDesc[0], self.gradInput:data());
    return self.gradInput
end

function SpatialFullConvolution:accGradParameters(input, gradOutput, scale)
    self.scaleT = self.scaleT or self.weight.new(1)
    -- this line forces this member to always be on CPU (needed for cudnn)
    self.scaleT = torch.type(self.weight) == 'torch.CudaDoubleTensor'
       and self.scaleT:double() or self.scaleT:float()
    scale = scale or 1.0
    self.scaleT[1] = scale

    assert(gradOutput:dim() == 3 or gradOutput:dim() == 4,
           'gradOutput has to be 3D or 4D');
    assert(gradOutput:isContiguous(), 'gradOutput has to be contiguous')
    if not self.weightDesc then self:resetWeightDescriptors() end
    self:createIODescriptors(input)

    -- gradBias
    if self.bias then
        errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
                 self.scaleT:data(),
                 self.oDescForBias[0], gradOutput:data(),
                 cudnn.scalar(input, 1),
                 self.biasDesc[0], self.gradBias:data())
    end

    -- gradWeight
    errcheck('cudnnConvolutionBackwardFilter', cudnn.getHandle(),
             self.scaleT:data(),
             self.oDesc[0], gradOutput:data(),
             self.iDesc[0], input:data(),
             self.convDesc[0],
             self.bwdFilterAlgType[0],
             self.extraBuffer:data(), self.extraBufferSizeInBytes,
             cudnn.scalar(input, 1),
             self.weightDesc[0], self.gradWeight:data())
end

function SpatialFullConvolution:clearDesc()
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
end

function SpatialFullConvolution:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function SpatialFullConvolution:clearState()
   self:clearDesc()
   return nn.Module.clearState(self)
end

function SpatialFullConvolution:read(file, version)
   parent.read(self, file)
   self.adjW = self.adjW or 0
   self.adjH = self.adjH or 0
end
