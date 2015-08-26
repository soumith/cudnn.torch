-- this file attempts to provide a purely functional set of bindings
-- all functions in this file retain absolutely no state.
-- There shouldn't be any reference to "self" in this file.

local cudnn = require 'cudnn.env'
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

cudnn.functional = {}

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

local function Batch2D(t)
    return t:view(1, t:size(1), t:size(2), t:size(3))
end

-- accumulates the bias into output.
-- output is assumed to be allocated and given.
cudnn.functional.bias2D_updateOutput = function(handle, bias, output)
    output = output:dim() == 3 and Batch2D(output) or output

    local biasDesc = cudnn.toDescriptor(bias:view(1, bias:nElement(),1,1))
    local oDesc = cudnn.toDescriptor(output)
    errcheck('cudnnAddTensor', handle,
             'CUDNN_ADD_SAME_C',
             one:data(), biasDesc[0], bias:data(),
             one:data(), oDesc[0], output:data())
end

-- accumulates the gradients into gradBias.
-- gradBias is assumed to be allocated and given.
cudnn.functional.bias2D_accGradParameters = function(handle, gradOutput, gradBias, scale)
    gradOutput = gradOutput:dim() == 3 and Batch2D(gradOutput) or gradOutput
    scale = scale or 1.0
    local scaleT = torch.FloatTensor({scale})
    local oDesc = cudnn.toDescriptor(gradOutput)
    local biasDesc = cudnn.toDescriptor(gradBias:view(1, gradBias:nElement(),1,1))
    errcheck('cudnnConvolutionBackwardBias', handle,
             scaleT:data(),
             oDesc[0], gradOutput:data(),
             one:data(),
             biasDesc[0], gradBias:data())
end

-- Does a 2D Convolution (updateOutput) on input, weight
-- output is assumed to be allocated and given.
cudnn.functional.Convolution2D_updateOutput = function(handle, input, weight, output,
                                                strideH, strideW, padH, padW, workspace)
    input = input:dim() == 3 and Batch2D(input) or input
    output = output:dim() == 3 and Batch2D(output) or output

    -- create a weight descriptor
    local weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
   errcheck('cudnnCreateFilterDescriptor', weightDesc)
   local nOutputPlane, nInputPlane, kH, kW
       = weight:size(1), weight:size(2), weight:size(3), weight:size(4)
   local desc = torch.IntTensor({nOutputPlane, nInputPlane, kH, kW})
   errcheck('cudnnSetFilterNdDescriptor', weightDesc[0], 'CUDNN_DATA_FLOAT', 4,
            desc:data());
   local function destroyWDesc(d)
      errcheck('cudnnDestroyFilterDescriptor', d[0]);
   end
   ffi.gc(weightDesc, destroyWDesc)

   -- create a convolution descriptor
   local convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
   errcheck('cudnnCreateConvolutionDescriptor', convDesc)
   local pad = torch.IntTensor({padH, padW})
   local stride = torch.IntTensor({strideH, strideW})
   local upscale = torch.IntTensor({1,1})
   errcheck('cudnnSetConvolutionNdDescriptor', convDesc[0],
            2, pad:data(),
            stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION');
   local function destroyConvDesc(d)
       errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
   end
   ffi.gc(convDesc, destroyConvDesc)

    -- create input descriptor
   local iDesc = cudnn.toDescriptor(input)

   -- create output descriptor
   local oSize = torch.IntTensor(4)
   errcheck('cudnnGetConvolutionNdForwardOutputDim',
            convDesc[0], iDesc[0],
            weightDesc[0], 4, oSize:data())
   oSize = oSize:long()
   assert(output:dim() == 4 and
          output:size(1) == oSize[1] and
          output:size(2) == oSize[2] and
          output:size(3) == oSize[3] and
          output:size(4) == oSize[4],
          'Output is of wrong size')
   -- create descriptor for output
   local oDesc = cudnn.toDescriptor(output)

   -- create forwardAlgorithm descriptors for
   local algType = ffi.new("cudnnConvolutionFwdAlgo_t[?]", 1)
   local algSearchMode = 'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT'
   local algWorkspaceLimit = 0
   if workspace then
       algWorkspaceLimit = workspace:nElement() * 4 -- 4 = sizeof float
   end
   errcheck('cudnnGetConvolutionForwardAlgorithm',
            handle,
            iDesc[0], weightDesc[0],
            convDesc[0], oDesc[0],
            algSearchMode, algWorkspaceLimit, algType)

   -- do convolution
   errcheck('cudnnConvolutionForward', handle,
            one:data(),
            iDesc[0], input:data(),
            weightDesc[0], weight:data(),
            convDesc[0], algType[0],
            workspace and workspace:data() or nil, algWorkspaceLimit,
            zero:data(),
            oDesc[0], output:data());
end

-- Does a 2D Convolution (updateGradInput) on input, weight, output, gradOutput
-- gradInput is assumed to be allocated and given.
cudnn.functional.Convolution2D_updateGradInput = function(handle, input, weight, output, gradOutput, gradInput,
                                                   strideH, strideW, padH, padW)
    input = input:dim() == 3 and Batch2D(input) or input
    output = output:dim() == 3 and Batch2D(output) or output
    gradOutput = gradOutput:dim() == 3 and Batch2D(gradOutput) or gradOutput
    gradInput = gradInput:dim() == 3 and Batch2D(gradInput) or gradInput

    -- create a weight descriptor
    local weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
   errcheck('cudnnCreateFilterDescriptor', weightDesc)
   local nOutputPlane, nInputPlane, kH, kW
       = weight:size(1), weight:size(2), weight:size(3), weight:size(4)
   local desc = torch.IntTensor({nOutputPlane, nInputPlane, kH, kW})
   errcheck('cudnnSetFilterNdDescriptor', weightDesc[0], 'CUDNN_DATA_FLOAT', 4,
            desc:data());
   local function destroyWDesc(d)
      errcheck('cudnnDestroyFilterDescriptor', d[0]);
   end
   ffi.gc(weightDesc, destroyWDesc)

   -- create a convolution descriptor
   local convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
   errcheck('cudnnCreateConvolutionDescriptor', convDesc)
   local pad = torch.IntTensor({padH, padW})
   local stride = torch.IntTensor({strideH, strideW})
   local upscale = torch.IntTensor({1,1})
   errcheck('cudnnSetConvolutionNdDescriptor', convDesc[0],
            2, pad:data(),
            stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION');
   local function destroyConvDesc(d)
       errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
   end
   ffi.gc(convDesc, destroyConvDesc)

    -- create input, output descriptor
   local iDesc = cudnn.toDescriptor(input)
   local oDesc = cudnn.toDescriptor(output)

   -- do convolution
   errcheck('cudnnConvolutionBackwardData', handle,
            one:data(),
            weightDesc[0], weight:data(),
            oDesc[0], gradOutput:data(),
            convDesc[0],
            zero:data(),
            iDesc[0], gradInput:data());
end

-- accumulates the gradients into gradWeight.
-- gradWeight is assumed to be allocated and given.
local scaleT = torch.FloatTensor(1):fill(1.0)
cudnn.functional.Convolution2D_accGradParameters = function(handle, input, gradWeight, gradOutput,
                                                   strideH, strideW, padH, padW, scale)
    input = input:dim() == 3 and Batch2D(input) or input
    gradOutput = gradOutput:dim() == 3 and Batch2D(gradOutput) or gradOutput

    scale = scale or 1.0
    scaleT[1] = scale
    -- create a weight descriptor
    local weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
    errcheck('cudnnCreateFilterDescriptor', weightDesc)
    local nOutputPlane, nInputPlane, kH, kW
        = gradWeight:size(1), gradWeight:size(2), gradWeight:size(3), gradWeight:size(4)
    local desc = torch.IntTensor({nOutputPlane, nInputPlane, kH, kW})
    errcheck('cudnnSetFilterNdDescriptor', weightDesc[0], 'CUDNN_DATA_FLOAT', 4,
             desc:data());
    local function destroyWDesc(d)
        errcheck('cudnnDestroyFilterDescriptor', d[0]);
    end
    ffi.gc(weightDesc, destroyWDesc)

    -- create a convolution descriptor
    local convDesc = ffi.new('struct cudnnConvolutionStruct*[1]')
    errcheck('cudnnCreateConvolutionDescriptor', convDesc)
    local pad = torch.IntTensor({padH, padW})
    local stride = torch.IntTensor({strideH, strideW})
    local upscale = torch.IntTensor({1,1})
    errcheck('cudnnSetConvolutionNdDescriptor', convDesc[0],
             2, pad:data(),
             stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION');
    local function destroyConvDesc(d)
        errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
    end
    ffi.gc(convDesc, destroyConvDesc)

    -- create input, output descriptor
    local iDesc = cudnn.toDescriptor(input)
    local oDesc = cudnn.toDescriptor(gradOutput)

    -- do convolution
    errcheck('cudnnConvolutionBackwardFilter', cudnn.getHandle(),
             scaleT:data(),
             iDesc[0], input:data(),
             oDesc[0], gradOutput:data(),
             convDesc[0],
             one:data(),
             weightDesc[0], gradWeight:data());
end



-- Does a 2D Pooling (updateOutput) on input, weight
-- output is assumed to be allocated and given.
cudnn.functional.Pooling_updateOutput = function(handle, mode, input, output,
                                          kH, kW, dH, dW, padH, padW, ceil_mode)
    input = input:dim() == 3 and Batch2D(input) or input
    output = output:dim() == 3 and Batch2D(output) or output

    padH = padH or 0
    padW = padW or 0
    ceil_mode = ceil_mode or false

    local oW, oH
    if ceil_mode then
        oW = math.ceil((input:size(4)+padW*2 - kW)/dW + 1)
        oH = math.ceil((input:size(3)+padH*2 - kH)/dH + 1)
    else
        oW = math.floor((input:size(4)+padW*2 - kW)/dW + 1)
        oH = math.floor((input:size(3)+padH*2 - kH)/dH + 1)
    end
    assert(oH == output:size(3) and oW == output:size(4),
           'size mismatch: ' .. oH .. 'x' .. oW .. ' vs ' ..
           output:size(3) .. 'x' .. output:size(4))

    -- create pooling descriptor
    local poolDesc = ffi.new('struct cudnnPoolingStruct*[1]')
    errcheck('cudnnCreatePoolingDescriptor', poolDesc)
    local ker = torch.IntTensor({kH, kW})
    local str = torch.IntTensor({dH, dW})
    local pad = torch.IntTensor({padH, padW})
    errcheck('cudnnSetPoolingNdDescriptor', poolDesc[0], mode, 2,
             ker:data(), pad:data(), str:data());
    local function destroyPoolDesc(d)
        errcheck('cudnnDestroyPoolingDescriptor', d[0]);
    end
    ffi.gc(poolDesc, destroyPoolDesc)

    -- create input, output descriptor
    local iDesc = cudnn.toDescriptor(input)
    local oDesc = cudnn.toDescriptor(output)

    -- pool
    errcheck('cudnnPoolingForward', handle,
             poolDesc[0],
             one:data(),
             iDesc[0], input:data(),
             zero:data(),
             oDesc[0], output:data());
end

cudnn.functional.MaxPooling2D_updateOutput = function(handle, input, output,
                                               kH, kW, dH, dW, padH, padW, ceil_mode)
    cudnn.functional.Pooling_updateOutput(handle, 'CUDNN_POOLING_MAX', input, output,
                                          kH, kW, dH, dW, padH, padW, ceil_mode);
end

cudnn.functional.AveragePooling2D_updateOutput = function(handle, input, output,
                                               kH, kW, dH, dW, padH, padW, ceil_mode)
    cudnn.functional.Pooling_updateOutput(handle, 'CUDNN_POOLING_AVERAGE', input, output,
                                          kH, kW, dH, dW, padH, padW, ceil_mode);
end

-- Does a 2D Pooling (updateGradInput) on input, weight
-- output is assumed to be allocated and given.
cudnn.functional.Pooling_updateGradInput = function(handle, mode, input, output, gradOutput, gradInput,
                                          kH, kW, dH, dW, padH, padW, ceil_mode)
    input = input:dim() == 3 and Batch2D(input) or input
    output = output:dim() == 3 and Batch2D(output) or output
    gradOutput = gradOutput:dim() == 3 and Batch2D(gradOutput) or gradOutput
    gradInput = gradInput:dim() == 3 and Batch2D(gradInput) or gradInput

    padH = padH or 0
    padW = padW or 0
    ceil_mode = ceil_mode or false

    local oW, oH
    if ceil_mode then
        oW = math.ceil((input:size(4)+padW*2 - kW)/dW + 1)
        oH = math.ceil((input:size(3)+padH*2 - kH)/dH + 1)
    else
        oW = math.floor((input:size(4)+padW*2 - kW)/dW + 1)
        oH = math.floor((input:size(3)+padH*2 - kH)/dH + 1)
    end
    assert(oH == output:size(3) and oW == output:size(4),
           'size mismatch: ' .. oH .. 'x' .. oW .. ' vs ' ..
           output:size(3) .. 'x' .. output:size(4))

    -- create pooling descriptor
    local poolDesc = ffi.new('struct cudnnPoolingStruct*[1]')
    errcheck('cudnnCreatePoolingDescriptor', poolDesc)
    local ker = torch.IntTensor({kH, kW})
    local str = torch.IntTensor({dH, dW})
    local pad = torch.IntTensor({padH, padW})
    errcheck('cudnnSetPoolingNdDescriptor', poolDesc[0], mode, 2,
             ker:data(), pad:data(), str:data());
    local function destroyPoolDesc(d)
        errcheck('cudnnDestroyPoolingDescriptor', d[0]);
    end
    ffi.gc(poolDesc, destroyPoolDesc)

    -- create input, output descriptor
    local iDesc = cudnn.toDescriptor(input)
    local oDesc = cudnn.toDescriptor(output)

    -- pool
    errcheck('cudnnPoolingBackward',
             handle, poolDesc[0],
             one:data(),
             oDesc[0], output:data(),
             oDesc[0], gradOutput:data(),
             iDesc[0], input:data(),
             zero:data(),
             iDesc[0], gradInput:data());
end

cudnn.functional.MaxPooling2D_updateGradInput = function(handle, input, output, gradOutput, gradInput,
                                               kH, kW, dH, dW, padH, padW, ceil_mode)
    cudnn.functional.Pooling_updateGradInput(handle, 'CUDNN_POOLING_MAX', input, output, gradOutput, gradInput,
                                          kH, kW, dH, dW, padH, padW, ceil_mode);
end

cudnn.functional.AveragePooling2D_updateGradInput = function(handle, input, output, gradOutput, gradInput,
                                               kH, kW, dH, dW, padH, padW, ceil_mode)
    cudnn.functional.Pooling_updateGradInput(handle, 'CUDNN_POOLING_AVERAGE', input, output, gradOutput, gradInput,
                                          kH, kW, dH, dW, padH, padW, ceil_mode);
end
