cudnn.torch
===========

Torch7 FFI bindings for NVIDIA cuDNN (R5) kernels!

Modules are API compatible their [`nn`](https://github.com/torch/nn) equivalents. Fully unit-tested against `nn` implementations.
Conversion between `nn` and `cudnn` is available through `cudnn.convert` function.

#### Installation

* Install cuDNN (version R5 EA)
* Have at least CUDA 7.0
* Have `libcudnn.so` in your library path ($LD_LIBRARY_PATH) (Install cuDNN it from https://developer.nvidia.com/cuDNN )
* Instead of the previous step, you can copy the library files into /usr/local/cuda/lib64/ or to the corresponding folders in CUDA directory

#### Modules

```lua
-- All inputs have to be 3D or 4D(batch-mode), except ReLU, Tanh, Sigmoid, and BatchNormalization
cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW = 1], [dH = 1], [padW = 0], [padH = 0], [groups = 1])
cudnn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
cudnn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)

-- the pointwise functions take an additional optional argument. if inplace=true then they do operations in-place without using any extra memory for themselves
cudnn.ReLU(inplace[=false])
cudnn.ClippedReLU(ceiling, inplace[=false])
cudnn.Tanh(inplace[=false])
cudnn.Sigmoid(inplace[=false])

-- SoftMax can be run in fast mode or accurate mode. Default is accurate mode.
cudnn.SoftMax(fastMode [= false])          -- SoftMax across each image (just like nn.SoftMax)
cudnn.LogSoftMax()                         -- LogSoftMax across each image (just like nn.LogSoftMax)
cudnn.SpatialSoftMax(fastMode [= false])   -- SoftMax across feature-maps (per spatial location)
cudnn.SpatialLogSoftMax()                  -- LogSoftMax across feature-maps (per spatial location)
cudnn.VolumetricSoftMax(fastMode [= false])   -- SoftMax across feature-maps (per spatial location)
cudnn.VolumetricLogSoftMax()                  -- LogSoftMax across feature-maps (per spatial location)

cudnn.SpatialCrossEntropyCriterion()       -- A spatial version of LogSoftMax + ClassNLLCriterion in one shot
cudnn.VolumetricCrossEntropyCriterion()       -- A volumetric version of LogSoftMax + ClassNLLCriterion in one shot

-- Batch Normalization
cudnn.BatchNormalization(nFeature, eps, momentum, affine) -- same arguments as https://github.com/torch/nn/blob/master/doc/simple.md#nn.BatchNormalization
cudnn.SpatialBatchNormalization(nFeature, eps, momentum, affine)
cudnn.VolumetricBatchNormalization(nFeature, eps, momentum, affine)


-- Volumetric inputs (4D or 5D batched mode)
cudnn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH)
cudnn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH)
cudnn.VolumetricAveragePooling(kT, kW, kH, dT, dW, dH, padT, padW, padH)

-- Recurrent Modules

-- All inputs have to be 3D. Accepts input of seqLength x batch x inputDim, or batch x seqLength x inputDim if batchFirst set to true.
cudnn.RNNReLU(inputDim, outputDim, numberOfLayers, [batchFirst = false])
cudnn.RNNTanh(inputDim, outputDim, numberOfLayers, [batchFirst = false])
cudnn.LSTM(inputDim, outputDim, numberOfLayers, [batchFirst = false])
cudnn.GRU(inputDim, outputDim, numberOfLayers, [batchFirst = false])
cudnn.BLSTM(inputDim, outputDim, numberOfLayers, [batchFirst = false])
```

### Modes
There are two globally availabe modes useful for tuning performance:
```lua
require 'cudnn'
cudnn.benchmark = true -- uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                       -- If this is set to false, uses some in-built heuristics that might not always be fastest.
```
by default `cudnn.benchmark` is set to `false`.  Setting to `true` will improve performance, at the expense of using more
memory.  The input shape should be the same for each batch, otherwise autotune will re-run for each batch,
causing a huge slow-down.

```lua
cudnn.fastest = true -- this is like the :fastest() mode for the Convolution modules,
                     -- simply picks the fastest convolution algorithm, rather than tuning for workspace size
```
by default, `cudnn.fastest` is set to `false`.  You should set to `true` if memory is not an issue, and you
want the fastest performance


```lua
cudnn.verbose = true -- this prints out some more verbose information useful for debugging
```
by default, `cudnn.verbose` is set to `false`.

### Conversion between `cudnn` and `nn`

Conversion is done by `cudnn.convert` function which takes a network and backend arguments and goes over
network modules recursively substituting equivalents. No memory copy is done, just metatables are swapped.
If you don't want to convert all modules you can pass a function as the third argument to `cudnn.convert`.
It will be called at each step, with a module that is currently converted.  It is meant to exclude
modules i.e. if it returns `true`, they will be left untouched, otherwise they will be subject to conversion.

`Note that you cannot do backward pass when using cuDNN and when your model has batch normalization layers and is in evaluate mode.`

```lua
net = nn.Sequential()
net:add(nn.SpatialConvolution(3,96,11,11,3,3))
net:add(nn.ReLU())
cudnn.convert(net, cudnn)
print(net)

net = nn.Sequential()
net:add(nn.SpatialConvolution(3,96,11,11,3,3))
net:add(nn.ReLU())
cudnn.convert(net, cudnn, function(module)
   return torch.type(module):find('ReLU')
end)
print(net)
```

will result in:
```
nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): cudnn.SpatialConvolution(3 -> 96, 11x11, 3,3)
  (2): cudnn.ReLU
}
nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): cudnn.SpatialConvolution(3 -> 96, 11x11, 3,3)
  (2): nn.ReLU
}
```

### Older versions
For version CuDNN R1, checkout the branch **R1**
For version CuDNN R2, checkout the branch **R2**
For version CuDNN R3, checkout the branch **R3**
For version CuDNN R4, checkout the branch **R4**
