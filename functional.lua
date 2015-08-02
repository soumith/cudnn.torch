-- this file attempts to provide a purely functional set of bindings
-- all functions in this file retain absolutely no state.
-- There shouldn't be any reference to "self" in this file.

local cudnn = require 'cudnn.env'
local errcheck = cudnn.errcheck

cudnn.functional = {}

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

-- accumulates the bias into output.
-- output is assumed to be allocated and given.
cudnn.functional.SpatialBias_updateOutput = function(bias, output)
    local biasDesc = cudnn.toDescriptor(bias:view(1, bias:nElement(),1,1))
    local oDesc = cudnn.toDescriptor(output)
    errcheck('cudnnAddTensor', cudnn.getHandle(),
             'CUDNN_ADD_SAME_C',
             one:data(), biasDesc[0], bias:data(),
             one:data(), oDesc[0], output:data())
end

-- accumulates the gradients into gradBias.
-- gradBias is assumed to be allocated and given.
cudnn.functional.SpatialBias_accGradParameters = function(gradOutput, gradBias, scale)
    scale = scale or 1.0
    local scaleT = torch.FloatTensor({scale})
    local oDesc = cudnn.toDescriptor(gradOutput)
    local biasDesc = cudnn.toDescriptor(gradBias:view(1, gradBias:nElement(),1,1))
    errcheck('cudnnConvolutionBackwardBias', cudnn.getHandle(),
             scaleT:data(),
             oDesc[0], gradOutput:data(),
             one:data(),
             biasDesc[0], gradBias:data())
end
