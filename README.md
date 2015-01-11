cudnn.torch
===========

Torch7 FFI bindings for NVidia CuDNN kernels!

Modules are API compatible their [`nn`](https://github.com/torch/nn) equivalents. Fully unit-tested against `nn` implementations.

#### Installation

* Install CuDNN
* Have at least Cuda 6.5
* Have `libcudnn.so` in your library path (Install it from https://developer.nvidia.com/cuDNN )

#### Modules

```lua
-- All inputs have to be 3D or 4D(batch-mode), except ReLU, Tanh and Sigmoid
cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, nGroups)
cudnn.SpatialMaxPooling(kW, kH, dW, dH)
cudnn.SpatialAveragePooling(kW, kH, dW, dH)
-- the pointwise functions take an additional optional argument. if inplace=true then they do operations in-place without using any extra memory for themselves
cudnn.ReLU(inplace[=false])
cudnn.Tanh(inplace[=false])
cudnn.Sigmoid(inplace[=false])

-- SoftMax can be run in fast mode or accurate mode. Default is accurate mode.
cudnn.SoftMax(fastMode [= false])          -- SoftMax across each image (just like nn.SoftMax)
cudnn.SpatialSoftMax(fastMode [= false])   -- SoftMax across feature-maps (per spatial location)
```

I have no time to support these, so please don't expect a quick response to filed github issues.
