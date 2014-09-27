cudnn.torch
===========

Torch7 FFI bindings for NVidia CuDNN kernels!

Modules are API compatible their nn equivalents. Fully unit-tested against nn implementations

* Install CuDNN
* Have at least Cuda 6.5
* Have libcudnn.so in your library path (Install it from https://developer.nvidia.com/cuDNN )

####Modules
```
cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
cudnn.SpatialMaxPooling(kW, kH, dW, dH)
cudnn.SpatialAveragePooling(kW, kH, dW, dH)
cudnn.ReLU()
cudnn.Tanh()
cudnn.Sigmoid()
```

I have no time to support these, so please dont expect a quick response to filed github issues.
