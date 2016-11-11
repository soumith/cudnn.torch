-- this file attempts to provide a purely functional set of bindings
-- all functions in this file retain absolutely no state.
-- There shouldn't be any reference to "self" in this file.

local cudnn = require 'cudnn.env'
local ffi = require 'ffi'
local errcheck = cudnn.errcheck
local find = require 'cudnn.find'
cudnn.functional = {}

local function getMathType(weight)
   local mathType = cudnn.configmap(torch.type(weight))
   if mathType == 'CUDNN_DATA_HALF' then
      -- explicitly set math type to fp32 to avoid possible failures with fp16 and exotic sizes
      -- this can be changed back when ported to find() as it has built-in fallback mechanism
      mathType = 'CUDNN_DATA_FLOAT'
   end
   return mathType
end

local function Batch2D(t)
    return t:view(1, t:size(1), t:size(2), t:size(3))
end

local function scalar(tensor, v)
   if v ~= 1 and v ~= 0 then
      local a = torch.type(tensor) == 'torch.CudaDoubleTensor'
            and torch.DoubleTensor({v}) or torch.FloatTensor({v})
      return a:data()
   else
      return cudnn.scalar(tensor, v)
   end
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
cudnn.functional.bias2D_accGradParameters = function(handle, gradOutput, gradBias, scale, alpha)
    gradOutput = gradOutput:dim() == 3 and Batch2D(gradOutput) or gradOutput
    local oDesc = cudnn.toDescriptor(gradOutput)
    local biasDesc = cudnn.toDescriptor(gradBias:view(1, gradBias:nElement(),1,1))
    errcheck('cudnnConvolutionBackwardBias', handle,
             scalar(gradBias, scale or 1),
             oDesc[0], gradOutput:data(),
             scalar(gradBias, alpha or 1),
             biasDesc[0], gradBias:data())
end


-- Does a 2D Convolution (updateOutput) on input, weight
-- output is assumed to be allocated and given.
cudnn.functional.Convolution2D_updateOutput = function(handle, input, weight, output,
                                                strideH, strideW, padH, padW, workspace)
    input = input:dim() == 3 and Batch2D(input) or input
    output = output:dim() == 3 and Batch2D(output) or output

    -- create a weight descriptor
   local nOutputPlane, nInputPlane, kH, kW
       = weight:size(1), weight:size(2), weight:size(3), weight:size(4)
   local weightDesc = cudnn.setFilterDescriptor(
      { dataType = cudnn.typemap[torch.type(input)],
        filterDimA = {nOutputPlane, nInputPlane, kH, kW}})

   -- create a convolution descriptor
   local convDescData = { padA = {padH, padW},
        filterStrideA = {strideH, strideW},
        dataType = getMathType(weight) }
   local convDesc = cudnn.setConvolutionDescriptor(convDescData);

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

   local layer = {
      convDescData = convDescData,
      convDesc = convDesc,
      weight = weight,
      nInputPlane = nInputPlane,
      nOutputPlane = nOutputPlane,
      kW = kW,
      kH = kH,
      pad = {padH, padW},
      stride = {strideH, strideW},
   }

   local finder = find.get()
   find:prepare(layer, input, output)
   local fwdAlgo = finder:forwardAlgorithm(layer, {iDesc[0], input, weightDesc[0],
                                                   weight, convDesc[0], oDesc[0], output})
   local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()

   -- do convolution
   errcheck('cudnnConvolutionForward', handle,
            cudnn.scalar(input, 1),
            iDesc[0], input:data(),
            weightDesc[0], weight:data(),
            convDesc[0], fwdAlgo,
            extraBuffer, extraBufferSize,
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
   local nOutputPlane, nInputPlane, kH, kW
       = weight:size(1), weight:size(2), weight:size(3), weight:size(4)
   local weightDesc = cudnn.setFilterDescriptor(
      { dataType = cudnn.typemap[torch.type(input)],
        filterDimA = {nOutputPlane, nInputPlane, kH, kW} })

   -- create a convolution descriptor
   local convDescData = { padA = {padH, padW},
                          filterStrideA = {strideH, strideW},
                          dataType = getMathType(weight)
                        }
   local convDesc = cudnn.setConvolutionDescriptor(convDescData);
    -- create input, output descriptor
   local iDesc = cudnn.toDescriptor(input)
   local oDesc = cudnn.toDescriptor(output)

   local layer = {
      convDescData = convDescData,
      convDesc = convDesc,
      weight = weight,
      nInputPlane = nInputPlane,
      nOutputPlane = nOutputPlane,
      kW = kW,
      kH = kH,
      pad = {padH, padW},
      stride = {strideH, strideW},
   }

   local finder = find.get()
   find:prepare(layer, input, output)
   local bwdDataAlgo = finder:backwardDataAlgorithm(layer, {weightDesc[0], weight, oDesc[0],
                                                    output, convDesc[0], iDesc[0], input})
   local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()

   -- do convolution
   errcheck('cudnnConvolutionBackwardData', handle,
               cudnn.scalar(input, 1),
               weightDesc[0], weight:data(),
               oDesc[0], gradOutput:data(),
               convDesc[0], bwdDataAlgo,
               extraBuffer, extraBufferSize,
               cudnn.scalar(input, 0),
               iDesc[0], gradInput:data());
end

-- accumulates the gradients into gradWeight.
-- gradWeight is assumed to be allocated and given.
cudnn.functional.Convolution2D_accGradParameters = function(handle, input, gradWeight, gradOutput,
                                                   strideH, strideW, padH, padW, scale, alpha)
    input = input:dim() == 3 and Batch2D(input) or input
    gradOutput = gradOutput:dim() == 3 and Batch2D(gradOutput) or gradOutput

    -- create a weight descriptor
    local nOutputPlane, nInputPlane, kH, kW
        = gradWeight:size(1), gradWeight:size(2), gradWeight:size(3), gradWeight:size(4)

    local weightDesc =  cudnn.setFilterDescriptor({ dataType = cudnn.typemap[torch.type(input)],
                                                    filterDimA = {nOutputPlane, nInputPlane, kH, kW}})
    -- create a convolution descriptor
    local convDescData = { padA = {padH, padW},
                           filterStrideA = {strideH, strideW},
                           dataType = getMathType(gradWeight) }
    local convDesc = cudnn.setConvolutionDescriptor(convDescData);

    -- create input, output descriptor
    local iDesc = cudnn.toDescriptor(input)
    local oDesc = cudnn.toDescriptor(gradOutput)

   local layer = {
      convDesc = convDesc,
      convDescData = convDescData,
      weight = gradWeight,
      nInputPlane = nInputPlane,
      nOutputPlane = nOutputPlane,
      kW = kW,
      kH = kH,
      pad = {padH, padW},
      stride = {strideH, strideW},
   }

   local finder = find.get()
   find:prepare(layer, input, gradOutput)
   local bwdFilterAlgo = finder:backwardFilterAlgorithm(layer, {iDesc[0], input, oDesc[0],
                                                                gradOutput, convDesc[0], weightDesc[0], gradWeight})
   local extraBuffer, extraBufferSize = cudnn.getSharedWorkspace()

    -- do convolution
    errcheck('cudnnConvolutionBackwardFilter', handle,
             scalar(gradWeight, scale or 1),
             iDesc[0], input:data(),
             oDesc[0], gradOutput:data(),
             convDesc[0], bwdFilterAlgo,
             extraBuffer, extraBufferSize,
             scalar(gradWeight, alpha or 1),
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
