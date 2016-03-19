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

for i, mode_desc in ipairs({
	{'Forward AutoTuned            ', nil},
	{'Forward implicit gemm        ', 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM'},
	{'Forward implicit precomp gemm', 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'},
	{'Forward gemm                 ', 'CUDNN_CONVOLUTION_FWD_ALGO_GEMM'},
	{'Forward FFT                  ', 'CUDNN_CONVOLUTION_FWD_ALGO_FFT'},
	{'Forward FFT tiling           ', 'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING'},
--	{'Forward Winograd             ', 'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD'} -- not supported for this size
}) do
   local title = mode_desc[1]
   local mode = mode_desc[2]

   benchSpatial(title, from, to, kH, kW, sH, sW, iH, iW, batchSize, mode)
end

function benchVolumetric(title, nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH, kT_input, kW_input, kH_input, nBatch, ...)
   local gconv = cudnn.VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH):setMode(...):fastest():cuda()
   local input = torch.zeros(nBatch, nInputPlane, kT_input, kW_input, kH_input):cuda()
   local output = gconv:forward(input)
   cutorch.synchronize()

   local t1 = torch.Timer()
   local output = gconv:forward(input)
   cutorch.synchronize()
   print(title .. ': ', nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH, kT_input, kW_input, kH_input, nBatch, t1:time().real)
end

print("cudnn.VolumetricConvolution")

for i, mode_desc in ipairs({
	{'Forward AutoTuned            ', nil},
	{'Forward implicit gemm        ', 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM'},
	{'Forward implicit precomp gemm', 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'},
--	{'Forward gemm                 ', 'CUDNN_CONVOLUTION_FWD_ALGO_GEMM'}, -- not supported for this size
--	{'Forward FFT                  ', 'CUDNN_CONVOLUTION_FWD_ALGO_FFT'}, -- not supported for this size
	{'Forward FFT tiling           ', 'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING'},
--	{'Forward Winograd             ', 'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD'} -- not supported for this size
}) do
   local title = mode_desc[1]
   local mode = mode_desc[2]

    benchVolumetric(title, 256, 256,  3,3,3,  1,1,1, 1,1,1,  8,  28,  28, 50, mode)
end

-- For reference, CuDNN Convolution modes
--[[
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6

    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0         = 0,  // non-deterministic
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1         = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT       = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3         = 3   // non-deterministic, algo0 with workspace

    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0          = 0, // non-deterministic
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1          = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT        = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD   = 4
    ]]--
