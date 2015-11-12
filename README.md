cudnn.torch
===========

Torch7 FFI bindings for NVidia CuDNN (R4) kernels!

Modules are API compatible their [`nn`](https://github.com/torch/nn) equivalents. Fully unit-tested against `nn` implementations.

#### Installation

* Install CuDNN (version R4 EA)
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

cudnn.SpatialCrossEntropyCriterion()       -- A spatial version of LogSoftMax + ClassNLLCriterion in one shot

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
For version CuDNN R3, checkout the branch **R3**


R4 Release Notes:
- Rather than resolving v3-v4 diffs, I have imported new cudnn.h with its entirety and converted comments and defines. This should be less error-prone.
- addTensor_v2 uses changed to new AddTensor API.

R4 TODO:
per-activation BN code needs to be added (new .lua similar to SpatialBN.lua, as per Andrei:
I believe we have at least one thing missing - per-activation BN (Torch implementation in nn.BatchNormalization.lua).
What I believe we have now is an integration of implementation for nn.SpatialBatchNormalization.lua

This is very similar to SpatialBatchNormalizaiton.lua but should use a different cudnnBatchNormalizationMode_t and tensor dimensions need to be adjusted accordingly.
For Spatial BN normalization is performed over N with 1CHW result and for per-activation it's done over NHW with 1C11 result.

Per-activation BN is only used after non-convolutional layers where spatially-invariant behavior is not expected.
