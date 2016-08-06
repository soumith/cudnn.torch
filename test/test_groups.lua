require 'ccn2'
require 'cudnn'

bs = 32
ni = 96
no = 128
imsize = 55
groups = 2
kW = 7
stride = 3

ccn2_conv = ccn2.SpatialConvolution(ni,no,kW,stride,0,groups):cuda()
cudnn_conv = cudnn.SpatialConvolution(ni,no,kW,kW,stride,stride,0,0,groups):cuda()

input = torch.randn(bs,ni,imsize,imsize):cuda()
input_tr = input:transpose(1,4):transpose(1,3):transpose(1,2):contiguous()

cudnn_conv.weight:copy(ccn2_conv.weight:t())
cudnn_conv.bias:copy(ccn2_conv.bias)


cudnn_output = cudnn_conv:forward(input)
ccn2_output = ccn2_conv:forward(input_tr):transpose(4,1):transpose(4,2):transpose(4,3):contiguous()

cudnn_gradOutput = torch.randn(#cudnn_conv.output):cuda()
ccn2_gradOutput = cudnn_gradOutput:transpose(1,4):transpose(1,3):transpose(1,2):contiguous()

cudnn_gradInput = cudnn_conv:backward(input, cudnn_gradOutput)
ccn2_gradInput = ccn2_conv:backward(input_tr, ccn2_gradOutput)
ccn2_gradInput = ccn2_gradInput:transpose(4,1):transpose(4,2):transpose(4,3):contiguous()

cudnn_gradWeight = cudnn_conv.gradWeight
ccn2_gradWeight = ccn2_conv.gradWeight:t()

assert((cudnn_output - ccn2_output):abs():max() < 1e-4)
assert((cudnn_gradInput - ccn2_gradInput):abs():max() < 1e-4)
assert((cudnn_gradWeight - ccn2_gradWeight):abs():max() < 5e-2)

print 'no assertions'
