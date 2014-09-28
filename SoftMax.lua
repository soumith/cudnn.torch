local SoftMax, parent = torch.class('cudnn.SoftMax', 'cudnn.SpatialSoftMax')
local ffi = require 'ffi'
local C = cudnn.C
local errcheck = cudnn.errcheck

function SoftMax:__init(fast)
   parent.__init(self, fast)
   self.mode = 'CUDNN_SOFTMAX_MODE_INSTANCE'
end
