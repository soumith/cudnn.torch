-- modules that can be converted to nn seamlessly
local layer_list = {
  'BatchNormalization',
  'SpatialBatchNormalization',
  'SpatialConvolution',
  'SpatialCrossMapLRN',
  'SpatialFullConvolution',
  'SpatialMaxPooling',
  'SpatialAveragePooling',
  'ReLU',
  'Tanh',
  'Sigmoid',
  'SoftMax',
  'LogSoftMax',
  'VolumetricBatchNormalization',
  'VolumetricConvolution',
  'VolumetricMaxPooling',
  'VolumetricAveragePooling',
}

local layer_isize = {
  {'SpatialConvolution', 4},
  {'SpatialCrossMapLRN', 4},
  {'SpatialFullConvolution', 4},
  {'SpatialMaxPooling', 4},
  {'SpatialAveragePooling', 4},
  {'SpatialDivisiveNormalization', 4},
  {'SoftMax', 4},
  {'LogSoftMax', 4},
  {'TemporalConvolution', 4},
  {'VolumetricConvolution', 5},
  {'VolumetricMaxPooling', 5},
  {'VolumetricAveragePooling', 5},
}

-- goes over a given net and converts all layers to dst backend
-- for example: net = cudnn.convert(net, cudnn)
function cudnn.convert(net, dst, exclusion_fn)
  return net:replace(function(x)
    if torch.type(x) == 'nn.gModule' then
      io.stderr:write('Warning: cudnn.convert does not work with nngraph yet. Ignoring nn.gModule')
      return x
    end
    local y = 0
    local src = dst == nn and cudnn or nn
    local src_prefix = src == nn and 'nn.' or 'cudnn.'
    local dst_prefix = dst == nn and 'nn.' or 'cudnn.'

    local function convert(v)
      local y = {}
      torch.setmetatable(y, dst_prefix..v)
      if v == 'ReLU' then y = dst.ReLU() end -- because parameters
      for k,u in pairs(x) do y[k] = u end
      if src == cudnn and x.clearDesc then x.clearDesc(y) end
      if src == cudnn and v == 'SpatialAveragePooling' then
        y.divide = true
        y.count_include_pad = v.mode == 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
      end
      return y
    end

    if exclusion_fn and exclusion_fn(x) then
      return x
    end
    local t = torch.typename(x)
    if t == 'nn.SpatialConvolutionMM' then
      y = convert('SpatialConvolution')
    elseif t == 'inn.SpatialCrossResponseNormalization' then
      y = convert('SpatialCrossMapLRN')
    else
      for i,v in ipairs(layer_list) do
        if t == src_prefix..v then
          y = convert(v)
        end
      end
    end
    if y == 0 then y = x end

    -- hacky code to initialize iSize
    if src == nn then
      t = torch.typename(y)
      if t == 'cudnn.SpatialBatchNormalization' or t == 'cudnn.VolumetricBatchNormalization' then
        y.iSize = torch.LongStorage(y.nDim):fill(0)
      else
        for i,v in ipairs(layer_isize) do
          if t == dst_prefix..v[1] then
            y.iSize = torch.LongStorage(v[2]):fill(0)
          end
        end
      end
    end
    return y
  end)
end
