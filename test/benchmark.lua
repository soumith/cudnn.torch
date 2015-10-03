require 'cudnn'
require 'torch'

function bench(title, nInputC, nOutputC, kH, kW, sH, sW, iH, iW, nBatch, ...)
   local m1 = cudnn.SpatialConvolution(nInputC,nOutputC,kW,kH, sW, sH):setMode(...):fastest():cuda()
   local i1 = torch.zeros(nBatch, nInputC, iH, iW):cuda()
   local o1 = m1:forward(i1)

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

-- just auto-tuned by cudnn with CUDNN_CONVOLUTION_FWD_PREFER_FASTEST mode
bench('Forward AutoTuned            ', from, to, kH, kW, sH, sW, iH, iW, batchSize)

bench('Forward implicit gemm        ', from, to, kH, kW, sH, sW, iH, iW, batchSize,
      'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
      'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
      'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0')

bench('Forward implicit precomp gemm', from, to, kH, kW, sH, sW, iH, iW, batchSize,
      'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM',
      'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
      'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0')

bench('Forward gemm                 ', from, to, kH, kW, sH, sW, iH, iW, batchSize,
      'CUDNN_CONVOLUTION_FWD_ALGO_GEMM',
      'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
      'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0')


bench('Forward FFT                  ', from, to, kH, kW, sH, sW, iH, iW, batchSize,
      'CUDNN_CONVOLUTION_FWD_ALGO_FFT',
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
