require 'cunn'
require 'cudnn'

local h=5
local w=5
local bsz=4
local from=4
local input = torch.randn(bsz,from,h,w):cuda()
local gradOutput = torch.randn(bsz,from,h,w):cuda()
local cbn = cudnn.SpatialBatchNormalization(bsz, 1e-3):cuda()
local gbn = nn.SpatialBatchNormalization(bsz, 1e-3):cuda()
local groundtruth = gbn:forward(input)
local rescuda = cbn:forward(input)
local resgrad = cbn:backward(input, gradOutput)
local groundgrad = gbn:backward(input, gradOutput)
local error = (rescuda:float() - groundtruth:float()):abs():max()
print("error",error)
error = (resgrad:float() - groundgrad:float()):abs():max()
print("error back",error)
