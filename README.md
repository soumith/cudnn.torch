cudnn.torch
===========

Torch7 FFI bindings for NVidia CuDNN (R3) kernels!

Modules are API compatible their [`nn`](https://github.com/torch/nn) equivalents. Fully unit-tested against `nn` implementations.

#### Installation

* Install CuDNN (version R3)
* Have at least Cuda 7.0
* Have `libcudnn.so` in your library path (Install it from https://developer.nvidia.com/cuDNN )

#### Modules

```lua
-- All inputs have to be 3D or 4D(batch-mode), except ReLU, Tanh and Sigmoid
cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW = 1], [dH = 1], [padW = 0], [padH = 0], [groups = 1])
cudnn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
cudnn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)

-- the pointwise functions take an additional optional argument. if inplace=true then they do operations in-place without using any extra memory for themselves
cudnn.ReLU(inplace[=false])
cudnn.Tanh(inplace[=false])
cudnn.Sigmoid(inplace[=false])

-- SoftMax can be run in fast mode or accurate mode. Default is accurate mode.
cudnn.SoftMax(fastMode [= false])          -- SoftMax across each image (just like nn.SoftMax)
cudnn.LogSoftMax()                         -- LogSoftMax across each image (just like nn.LogSoftMax)
cudnn.SpatialSoftMax(fastMode [= false])   -- SoftMax across feature-maps (per spatial location)
cudnn.SpatialLogSoftMax()                  -- LogSoftMax across feature-maps (per spatial location)

-- Volumetric inputs (4D or 5D batched mode)
cudnn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH)
cudnn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH)
cudnn.VolumetricAveragePooling(kT, kW, kH, dT, dW, dH, padT, padW, padH)
```

### Modes
There are two globally availabe modes useful for tuning performance:
```lua
require 'cudnn'
cudnn.benchmark = true -- uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                       -- If this is set to false, uses some in-built heuristics that might not always be fastest.
```
by default `cudnn.benchmark` is set to `false`.

```lua
cudnn.fastest = true -- this is like the :fastest() mode for the Convolution modules,
                     -- simply picks the fastest convolution algorithm, rather than tuning for workspace size
```
by default, `cudnn.fastest` is set to `false`.


```lua
cudnn.verbose = true -- this prints out some more verbose information useful for debugging
```
by default, `cudnn.verbose` is set to `false`.


### Older versions
For version CuDNN R1, checkout the branch **R1**
For version CuDNN R2, checkout the branch **R2**
