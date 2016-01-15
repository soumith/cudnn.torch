-- modules that can be converted to nn seamlessly
local layer_list = {
  'SpatialConvolution',
  'SpatialCrossMapLRN',
  'SpatialMaxPooling',
  'SpatialAveragePooling',
  'ReLU',
  'Tanh',
  'Sigmoid',
  'SoftMax',
  'LogSoftMax',
  'VolumetricConvolution',
  'VolumetricMaxPooling',
  'VolumetricAveragePooling',
}

-- similar to nn.Module.apply
-- goes over a net and recursively replaces modules
-- using callback function
local function replace(self, callback)
  local out = callback(self)
  if self.modules then
    for i, module in ipairs(self.modules) do
      self.modules[i] = replace(module, callback)
    end
  end
  return out
end

-- goes over a given net and converts all layers to dst backend
-- for example: net = cudnn.convert(net, cudnn)
function cudnn.convert(net, dst)
  return replace(net, function(x)
    local y = 0
    local src = dst == nn and cudnn or nn
    local src_prefix = src == nn and 'nn.' or 'cudnn.'
    local dst_prefix = dst == nn and 'nn.' or 'cudnn.'

    local function convert(v)
      local y = {}
      torch.setmetatable(y, dst_prefix..v)
      if v == 'ReLU' then y = dst.ReLU() end -- because parameters
      for k,u in pairs(x) do y[k] = u end
      if src == cudnn and x.clearDesc then x:clearDesc() end
      return y
    end
    local t = torch.typename(x)
    if t == 'nn.SpatialConvolutionMM' then
      y = convert('SpatialConvolution')
    elseif t == 'inn.SpatialCrossResponseNormalization' then
      y = convert('SpatialCrossMapLRN')
    else
      for i,v in ipairs(layer_list) do
        if torch.typename(x) == src_prefix..v then
          y = convert(v)
        end
      end
    end
    return y == 0 and x or y
  end)
end

