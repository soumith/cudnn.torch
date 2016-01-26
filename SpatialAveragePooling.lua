local SpatialAveragePooling, parent
= torch.class('cudnn.SpatialAveragePooling', 'cudnn._Pooling')

local function backwardCompatible(self)
   if self.ceil_mode == nil then
      self.ceil_mode = false
      self.count_include_pad = true
      self.padH = 0
      self.padW = 0
   end
end

function SpatialAveragePooling:updateOutput(input)
   -- for nn <> cudnn conversion
   backwardCompatible(self)
   if self.divide ~= nil then
      assert(self.divide, 'not supported')
   end

   self.count_include_pad = self.count_include_pad ~= nil and
      self.count_include_pad or true
   if self.count_include_pad then
      self.mode = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
   else
      error'This mode is untested in cudnn'
      self.mode = 'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING'
   end
   return parent.updateOutput(self, input)
end

function SpatialAveragePooling:__tostring__()
   return nn.SpatialAveragePooling.__tostring__(self)
end
