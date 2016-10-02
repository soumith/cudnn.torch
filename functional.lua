-- this file attempts to provide a purely functional set of bindings
-- all functions in this file retain absolutely no state.
-- There shouldn't be any reference to "self" in this file.

local cudnn = require 'cudnn.env'
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local NULL
if not jit then
    NULL = ffi.C.NULL
end

cudnn.functional = {}




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
             cudnn.scalar(output, 1), biasDesc[0], bias:data(),
             cudnn.scalar(output, 1), oDesc[0], output:data())
end

-- accumulates the gradients into gradBias.
-- gradBias is assumed to be allocated and given.
cudnn.functional.bias2D_accGradParameters = function(handle, gradOutput, gradBias, scale)
    gradOutput = gradOutput:dim() == 3 and Batch2D(gradOutput) or gradOutput
    scale = scale or 1.0
    local scaleT = torch.type(gradBias) == 'torch.CudaDoubleTensor'
       and torch.DoubleTensor({scale}) or torch.FloatTensor({scale})
    local oDesc = cudnn.toDescriptor(gradOutput)
    local biasDesc = cudnn.toDescriptor(gradBias:view(1, gradBias:nElement(),1,1))
    errcheck('cudnnConvolutionBackwardBias', handle,
             scaleT:data(),
             oDesc[0], gradOutput:data(),
             cudnn.scalar(gradOutput, 1),
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
   errcheck('cudnnSetFilterNdDescriptor', weightDesc[0], cudnn.typemap[torch.type(input)], 'CUDNN_TENSOR_NCHW', 4,
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
            stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
            cudnn.configmap(torch.type(weight)));
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
   if cudnn.fastest then
      algSearchMode = 'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'
   end
   local algWorkspaceLimit = nInputPlane * kH * kW * cudnn.sizeof(weight)
   
   errcheck('cudnnGetConvolutionForwardAlgorithm',
            handle,
            iDesc[0], weightDesc[0],
            convDesc[0], oDesc[0],
            algSearchMode, algWorkspaceLimit, algType)

   local bufSize = torch.LongTensor(1)
   errcheck('cudnnGetConvolutionForwardWorkspaceSize',
              handle,
              iDesc[0], weightDesc[0],
              convDesc[0], oDesc[0],
              algType[0], bufSize:data())
   local maxBufSize = bufSize[1]

   local extraBuffer = workspace or cudnn.getSharedWorkspace()
   local extraBufferSizeInBytes = extraBuffer:nElement() * extraBuffer:elementSize()
   if maxBufSize > extraBufferSizeInBytes then
      extraBuffer:resize(math.ceil(maxBufSize / extraBuffer:elementSize()))
      extraBufferSizeInBytes = maxBufSize
   end

   -- do convolution
   errcheck('cudnnConvolutionForward', handle,
            cudnn.scalar(input, 1),
            iDesc[0], input:data(),
            weightDesc[0], weight:data(),
            convDesc[0], algType[0],
            extraBuffer:data(), extraBufferSizeInBytes,
            cudnn.scalar(input, 0),
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
   errcheck('cudnnSetFilterNdDescriptor', weightDesc[0], cudnn.typemap[torch.type(input)], 'CUDNN_TENSOR_NCHW', 4,
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
            stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
            cudnn.configmap(torch.type(weight)));
   local function destroyConvDesc(d)
       errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
   end
   ffi.gc(convDesc, destroyConvDesc)

    -- create input, output descriptor
   local iDesc = cudnn.toDescriptor(input)
   local oDesc = cudnn.toDescriptor(output)

   local algType = ffi.new("cudnnConvolutionBwdDataAlgo_t[?]", 1)
   local algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE'
   if cudnn.fastest then
      algSearchMode = 'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST'
   end
   local algWorkspaceLimit = nInputPlane * kH * kW * cudnn.sizeof(weight)

   errcheck('cudnnGetConvolutionBackwardDataAlgorithm',
            handle,
            weightDesc[0], oDesc[0],
            convDesc[0], iDesc[0],
            algSearchMode, algWorkspaceLimit, algType)

   local bufSize = torch.LongTensor(1)
   errcheck('cudnnGetConvolutionBackwardDataWorkspaceSize',
           handle,
           weightDesc[0], oDesc[0],
           convDesc[0], iDesc[0],
           algType[0], bufSize:data())
   local maxBufSize = bufSize[1]

   local extraBuffer = cudnn.getSharedWorkspace()
   local extraBufferSizeInBytes = extraBuffer:nElement() * extraBuffer:elementSize()
   if maxBufSize > extraBufferSizeInBytes then
      extraBuffer:resize(math.ceil(maxBufSize / extraBuffer:elementSize()))
      extraBufferSizeInBytes = maxBufSize
   end

   -- do convolution
   errcheck('cudnnConvolutionBackwardData', handle,
               cudnn.scalar(input, 1),
               weightDesc[0], weight:data(),
               oDesc[0], gradOutput:data(),
               convDesc[0],
               algType[0],
               extraBuffer:data(), extraBufferSizeInBytes,
               cudnn.scalar(input, 0),
               iDesc[0], gradInput:data());
end

-- accumulates the gradients into gradWeight.
-- gradWeight is assumed to be allocated and given.
cudnn.functional.Convolution2D_accGradParameters = function(handle, input, gradWeight, gradOutput,
                                                   strideH, strideW, padH, padW, scale)
    input = input:dim() == 3 and Batch2D(input) or input
    gradOutput = gradOutput:dim() == 3 and Batch2D(gradOutput) or gradOutput

    scale = scale or 1.0
    local scaleT = torch.type(gradWeight) == 'torch.CudaDoubleTensor'
       and torch.DoubleTensor({scale}) or torch.FloatTensor({scale})
    -- create a weight descriptor
    local weightDesc = ffi.new('struct cudnnFilterStruct*[1]')
    errcheck('cudnnCreateFilterDescriptor', weightDesc)
    local nOutputPlane, nInputPlane, kH, kW
        = gradWeight:size(1), gradWeight:size(2), gradWeight:size(3), gradWeight:size(4)
    local desc = torch.IntTensor({nOutputPlane, nInputPlane, kH, kW})
    errcheck('cudnnSetFilterNdDescriptor', weightDesc[0], cudnn.typemap[torch.type(input)], 'CUDNN_TENSOR_NCHW', 4,
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
             stride:data(), upscale:data(), 'CUDNN_CROSS_CORRELATION',
             cudnn.configmap(torch.type(gradWeight)));
    local function destroyConvDesc(d)
        errcheck('cudnnDestroyConvolutionDescriptor', d[0]);
    end
    ffi.gc(convDesc, destroyConvDesc)

    -- create input, output descriptor
    local iDesc = cudnn.toDescriptor(input)
    local oDesc = cudnn.toDescriptor(gradOutput)

    local algType = ffi.new("cudnnConvolutionBwdFilterAlgo_t[?]", 1)
    local algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE'
    if cudnn.fastest then
       algSearchMode = 'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST'
    end
    local algWorkspaceLimit = nInputPlane * kH * kW * cudnn.sizeof(gradWeight)

    errcheck('cudnnGetConvolutionBackwardFilterAlgorithm',
             handle,
             iDesc[0], oDesc[0],
             convDesc[0], weightDesc[0],
             algSearchMode, algWorkspaceLimit, algType)

    local bufSize = torch.LongTensor(1)
    errcheck('cudnnGetConvolutionBackwardFilterWorkspaceSize',
              handle,
              iDesc[0], oDesc[0],
              convDesc[0], weightDesc[0],
              algType[0], bufSize:data())
    local maxBufSize = bufSize[1]

    local extraBuffer = cudnn.getSharedWorkspace()
    local extraBufferSizeInBytes = extraBuffer:nElement() * extraBuffer:elementSize()
    if maxBufSize > extraBufferSizeInBytes then
       extraBuffer:resize(math.ceil(maxBufSize / extraBuffer:elementSize()))
       extraBufferSizeInBytes = maxBufSize
    end

    -- do convolution
    errcheck('cudnnConvolutionBackwardFilter', handle,
             scaleT:data(),
             iDesc[0], input:data(),
             oDesc[0], gradOutput:data(),
             convDesc[0],
             algType[0],
             extraBuffer:data(), extraBufferSizeInBytes,
             cudnn.scalar(input, 1),
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
    errcheck('cudnnSetPoolingNdDescriptor', poolDesc[0], mode, 'CUDNN_PROPAGATE_NAN', 2,
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
             cudnn.scalar(input, 1),
             iDesc[0], input:data(),
             cudnn.scalar(input, 0),
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
    errcheck('cudnnSetPoolingNdDescriptor', poolDesc[0], mode, 'CUDNN_PROPAGATE_NAN', 2,
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
             cudnn.scalar(input, 1),
             oDesc[0], output:data(),
             oDesc[0], gradOutput:data(),
             iDesc[0], input:data(),
             cudnn.scalar(input, 0),
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

local function createPointwiseDescriptors(mode, input, output)
   local activDesc = ffi.new('struct cudnnActivationStruct*[1]')
   errcheck('cudnnCreateActivationDescriptor', activDesc)
   errcheck('cudnnSetActivationDescriptor', activDesc[0], mode, 'CUDNN_PROPAGATE_NAN', 0.0);

   local function destroyADesc(a)
      if (a[0]) then
         errcheck('cudnnDestroyActivationDescriptor', a[0]);
         a[0] = nil
      end
   end
   ffi.gc(activDesc, destroyADesc)

   local iDesc = cudnn.toDescriptor(input:view(1,1,1,-1))
   return activDesc, iDesc
end

local function pointwise_updateOutput(handle, mode, input, output)
   local activDesc, iDesc = createPointwiseDescriptors(mode, input, output)
   errcheck('cudnnActivationForward',
            handle, activDesc[0],
            cudnn.scalar(input, 1),
            iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            iDesc[0], output:data());
end

local function pointwise_updateGradInput(handle, mode, input, output, gradOutput, gradInput)
   local activDesc, iDesc = createPointwiseDescriptors(mode, input, output)
   errcheck('cudnnActivationBackward',
            handle, activDesc[0],
            cudnn.scalar(input, 1),
            iDesc[0], output:data(),
            iDesc[0], gradOutput:data(),
            iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            iDesc[0], gradInput:data());
end

cudnn.functional.ReLU_updateOutput = function(handle, input, output)
   output:resizeAs(input)
   pointwise_updateOutput(handle, 'CUDNN_ACTIVATION_RELU', input, output)
end

cudnn.functional.ReLU_updateGradInput = function(handle, input, output, gradOutput, gradInput)
   gradInput:resizeAs(input)
   pointwise_updateGradInput(handle, 'CUDNN_ACTIVATION_RELU', input, output, gradOutput, gradInput)
end

cudnn.functional.Tanh_updateOutput = function(handle, input, output)
   output:resizeAs(input)
   pointwise_updateOutput(handle, 'CUDNN_ACTIVATION_TANH', input, output)
end

cudnn.functional.Tanh_updateGradInput = function(handle, input, output, gradOutput, gradInput)
   gradInput:resizeAs(input)
   pointwise_updateGradInput(handle, 'CUDNN_ACTIVATION_TANH', input, output, gradOutput, gradInput)
end

cudnn.functional.Sigmoid_updateOutput = function(handle, input, output)
   output:resizeAs(input)
   pointwise_updateOutput(handle, 'CUDNN_ACTIVATION_SIGMOID', input, output)
end

cudnn.functional.Sigmoid_updateGradInput = function(handle, input, output, gradOutput, gradInput)
   gradInput:resizeAs(input)
   pointwise_updateGradInput(handle, 'CUDNN_ACTIVATION_SIGMOID', input, output, gradOutput, gradInput)
end


local function softmax_updateOutput(handle, mode, algorithm, input, output)
   output:resizeAs(input)
   local iDesc = cudnn.toDescriptor(input)
   local oDesc = cudnn.toDescriptor(output)
   errcheck('cudnnSoftmaxForward',
            handle,
            mode, algorithm,
            cudnn.scalar(input, 1),
            iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            oDesc[0], output:data());
end

local function softmax_updateGradInput(handle, mode, algorithm, input, output, gradOutput, gradInput)
   gradInput:resizeAs(input)
   local iDesc = cudnn.toDescriptor(input)
   local oDesc = cudnn.toDescriptor(output)
   errcheck('cudnnSoftmaxBackward',
            handle,
            mode, algorithm,
            cudnn.scalar(input, 1),
            oDesc[0], output:data(),
            oDesc[0], gradOutput:data(),
            cudnn.scalar(input, 0),
            iDesc[0], gradInput:data());
end

cudnn.functional.LogSoftMax_updateOutput = function(handle, input, output)
   softmax_updateOutput(handle,
                        'CUDNN_SOFTMAX_LOG',
                        'CUDNN_SOFTMAX_MODE_INSTANCE',
                        input, output)
end

cudnn.functional.LogSoftMax_updateGradInput = function(handle, input, output, gradOutput, gradInput)
   softmax_updateGradInput(handle,
                           'CUDNN_SOFTMAX_LOG',
                           'CUDNN_SOFTMAX_MODE_INSTANCE',
                           input, output, gradOutput, gradInput)
end

cudnn.functional.SoftMax_updateOutput = function(handle, input, output)
   softmax_updateOutput(handle,
                        'CUDNN_SOFTMAX_ACCURATE',
                        'CUDNN_SOFTMAX_MODE_INSTANCE',
                        input, output)
end

cudnn.functional.SoftMax_updateGradInput = function(handle, input, output, gradOutput, gradInput)
   softmax_updateGradInput(handle,
                           'CUDNN_SOFTMAX_ACCURATE',
                           'CUDNN_SOFTMAX_MODE_INSTANCE',
                           input, output, gradOutput, gradInput)
end

