cudnn.torch
===========

Torch7 FFI bindings for NVidia CuDNN kernels!

Modules are API compatible their nn equivalents. Fully unit-tested against nn implementations

* Install CuDNN
* Have at least Cuda 6.5
* Have libcudnn.so in your library path (Install it from https://developer.nvidia.com/cuDNN )

####Modules
```
-- All inputs have to be 4D, even for ReLU, SoftMax etc.
cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
cudnn.SpatialMaxPooling(kW, kH, dW, dH)
cudnn.SpatialAveragePooling(kW, kH, dW, dH)
cudnn.ReLU()            
cudnn.Tanh()            
cudnn.Sigmoid()         

-- SoftMax can be run in fast mode or accurate mode. Default is accurate mode.
cudnn.SoftMax(fastMode [= false])          -- SoftMax across each image (just like nn.SoftMax)
cudnn.SpatialSoftMax(fastMode [= false])   -- SoftMax across feature-maps (per spatial location)
```

I have no time to support these, so please dont expect a quick response to filed github issues.
