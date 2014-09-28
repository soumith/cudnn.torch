local SpatialAveragePooling, parent
   = torch.class('cudnn.SpatialAveragePooling', 'cudnn.SpatialMaxPooling')
local ffi = require 'ffi'
local C = cudnn.C
local errcheck = cudnn.errcheck

function SpatialAveragePooling:__init(kW, kH, dW, dH)
   parent.__init(self, kW, kH, dW, dH)
end

function SpatialAveragePooling:resetPoolDescriptors()
   -- create pooling descriptor
   self.poolDesc = ffi.new('struct cudnnPoolingStruct*[1]')
   errcheck('cudnnCreatePoolingDescriptor', self.poolDesc)
   errcheck('cudnnSetPoolingDescriptor', self.poolDesc[0],
            'CUDNN_POOLING_AVERAGE',
            self.kH, self.kW, self.dH, self.dW);
   local function destroyPoolDesc(d)
      errcheck('cudnnDestroyPoolingDescriptor', d[0]);
   end
   ffi.gc(self.poolDesc, destroyPoolDesc)
end
