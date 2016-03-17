require 'cudnn'
require 'torch'

function benchSpatial(title, nInputC, nOutputC, kH, kW, sH, sW, iH, iW, nBatch, ...)
   local m1 = cudnn.SpatialConvolution(nInputC,nOutputC,kW,kH, sW, sH):setMode(...):fastest():cuda()
   local i1 = torch.zeros(nBatch, nInputC, iH, iW):cuda()
   local o1 = m1:forward(i1)
   cutorch.synchronize()

   local t1 = torch.Timer()
   local o1 = m1:forward(i1)
   cutorch.synchronize()
   print(title .. ': ', nInputC, nOutputC, kH, kW, iH, iW, nBatch, t1:time().real)
end


batchSize = 29
from = 14
to = 13
kW = 9
kH = 15
sW = 1
sH = 1
outW = 10
outH = 34
iW = (outW-1)*sW+kW
iH = (outH-1)*sH+kH


print('CUDNN Version: ', tonumber(cudnn.C.cudnnGetVersion()))
print("cudnn.SpatialConvolution")

-- just auto-tuned by cudnn with CUDNN_CONVOLUTION_FWD_PREFER_FASTEST mode
benchSpatial('Forward AutoTuned            ', from, to, kH, kW, sH, sW, iH, iW, batchSize)

benchSpatial('Forward implicit gemm        ', from, to, kH, kW, sH, sW, iH, iW, batchSize,
      'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
      'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
      'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0')

benchSpatial('Forward implicit precomp gemm', from, to, kH, kW, sH, sW, iH, iW, batchSize,
      'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM',
      'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
      'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0')

benchSpatial('Forward gemm                 ', from, to, kH, kW, sH, sW, iH, iW, batchSize,
      'CUDNN_CONVOLUTION_FWD_ALGO_GEMM',
      'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
      'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0')


benchSpatial('Forward FFT                  ', from, to, kH, kW, sH, sW, iH, iW, batchSize,
      'CUDNN_CONVOLUTION_FWD_ALGO_FFT',
      'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
      'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0')


function benchVolumetric(title, nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH, kT_input, kW_input, kH_input, nBatch, ...)
   local gconv = cudnn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH):setMode(...):fastest():cuda()
   local input = torch.zeros(nBatch, nInputPlane, kT_input, kW_input, kH_input):cuda()
   local output = gconv:forward(input)
   cutorch.synchronize()

   local t1 = torch.Timer()
   local output = gconv:forward(input)
   print(title .. ': ', nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH, kT_input, kW_input, kH_input, nBatch, t1:time().real)
end

print("cudnn.VolumetricConvolution")
benchVolumetric("Forward Autotuned            ",   3,  64,  3,3,3,  1,1,1, 1,1,1, 16, 112, 112, 50)
benchVolumetric("Forward Autotuned            ",  64,  64,  3,3,3,  1,1,1, 1,1,1, 16,  56,  56, 50)
benchVolumetric("Forward Autotuned            ", 128, 128,  3,3,3,  1,1,1, 1,1,1,  8,  28,  28, 50)
benchVolumetric("Forward Autotuned            ", 256, 256,  3,3,3,  1,1,1, 1,1,1,  8,  28,  28, 50)
benchVolumetric("Forward Autotuned            ", 256, 256,  3,3,3,  1,1,1, 1,1,1,  4,  14,  14, 50)
benchVolumetric("Forward Autotuned            ", 512, 512,  3,3,3,  1,1,1, 1,1,1,  4,  14,  14, 50)
benchVolumetric("Forward Autotuned            ", 512, 512,  3,3,3,  1,1,1, 1,1,1,  2,   7,   7, 50)

benchVolumetric("Forward Autotuned            ", 512, 512,  3,3,3,  1,1,1, 1,1,1,  2,   7,   7, 50)

benchVolumetric("Forward implicit gemm        ", 512, 512,  3,3,3,  1,1,1, 1,1,1,  2,   7,   7, 50,
      'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
      'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
      'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0')

benchVolumetric("Forward implicit precomp gemm", 512, 512,  3,3,3,  1,1,1, 1,1,1,  2,   7,   7, 50,
      'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM',
      'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
      'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0')


-- For reference, CuDNN Convolution modes
--[[
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3, // Placeholder
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4

    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0         = 0,  // non-deterministic
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1         = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT       = 2

    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0         = 0, // non-deterministic
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1         = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT       = 2,

    ]]--
