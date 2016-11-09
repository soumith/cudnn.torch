local TemporalConvolution, parent =
    torch.class('cudnn.TemporalConvolution', 'nn.TemporalConvolution')
--use cudnn to perform temporal convolutions
--note: if padH parameter is not passed, no padding will be performed, as in parent TemporalConvolution
--however, instead of separately padding data, as is required now for nn.TemporalConvolution,
--it is recommended to pass padding parameter to this routine and use cudnn implicit padding facilities.
--limitation is that padding will be equal on both sides.

local Convolution = cudnn.SpatialConvolution

function TemporalConvolution:__init(inputFrameSize, outputFrameSize,
                            kH, dH, padH)
    local delayedReset = self.reset
    local kW = inputFrameSize
    local nInputPlane = 1 -- single channel
    local nOutputPlane = outputFrameSize
    self.inputFrameSize = inputFrameSize
    self.outputFrameSize = outputFrameSize
    Convolution.__init(self, nInputPlane, nOutputPlane, kW, kH, 1, dH,0,padH)
    self.weight = self.weight:view(nOutputPlane,inputFrameSize*kH)
    self.gradWeight = self.gradWeight:view(outputFrameSize, inputFrameSize*kH)
--self.dW and self.kW now have different meaning than in nn.TemporalConvolution, because
--W and H are switched in temporal and spatial
end

function TemporalConvolution:createIODescriptors(input)
    local sizeChanged = false
    if not self.iDesc or not self.oDesc or
        input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
    or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
       sizeChanged = true
    end
    cudnn.SpatialConvolution.createIODescriptors(self,input)
    if sizeChanged then
       self.oSize = self.output:size()
    end
end

function TemporalConvolution:fastest(mode)
    cudnn.SpatialConvolution.fastest(self,mode)
    return self
end

function TemporalConvolution:setMode(fmode, bdmode, bwmode)
   return Convolution.setMode(self, fmode, bdmode, bwmode)
end

function TemporalConvolution:resetMode()
   return Convolution.resetMode(self)
end

function TemporalConvolution:resetWeightDescriptors()
    return Convolution.resetWeightDescriptors(self)
end

local function inputview(input)
   local _input = input
   if input:dim()==2 then
      _input = input:view(1,input:size(1),input:size(2))
   end
   return _input:view(_input:size(1),1,_input:size(2),_input:size(3))
end

function TemporalConvolution:updateOutput(input)
   local _input = inputview(input)
   assert(_input:size(4) == self.inputFrameSize,'invalid input frame size')
   self.buffer = self.buffer or input.new()
   self._output = self._output or input.new()
   if self.output:storage() then self._output:set(self.output:storage()) else self._output = self.output end
   if self.buffer:storage() then self.output:set(self.buffer:storage(), 1, self.output:size()) else self.output = self.buffer end
   Convolution.updateOutput(self, _input)
   self.buffer = self.output:view(self.oSize):transpose(2,3)
   self.output  = self._output:resize(self.buffer:size()):copy(self.buffer)
   -- self.output here is always 4D, use input dimensions to properly view output
   if input:dim()==3 then
     self.output=self.output:view(self.oSize[1], self.oSize[3],self.oSize[2])
   else
     self.output=self.output:view(self.oSize[3], self.oSize[2])
   end
   return self.output
end

local function transposeGradOutput(src,dst)
    assert(src:dim() == 2 or src:dim() == 3, 'gradOutput has to be 2D or 3D');
    local srctransposed = src:transpose(src:dim(),src:dim()-1)
    dst:resize(srctransposed:size())
    dst:copy(srctransposed)
    if src:dim()==3 then
      dst = dst:view(dst:size(1),dst:size(2),dst:size(3),1)
    else
      dst = dst:view(dst:size(1),dst:size(2),1)
    end
    return dst
end

function TemporalConvolution:updateGradInput(input, gradOutput)
   if not self.gradInput then return end
   local _gradOutput = transposeGradOutput(gradOutput,self.buffer)
   local _input = inputview(input)
   self.gradInput = Convolution.updateGradInput(self, _input, _gradOutput)
   if input:dim()==3 then
      self.gradInput = self.gradInput:view(self.iSize[1],self.iSize[3],self.iSize[4])
   else
      self.gradInput = self.gradInput:view(self.iSize[3],self.iSize[4])
   end
   return self.gradInput
end

function TemporalConvolution:accGradParameters(input,gradOutput,scale)
--2d (4d) view of input
    local _input = inputview(input)
-- transpose gradOutput (it will likely be transposed twice, hopefully, no big deal
    local _gradOutput = transposeGradOutput(gradOutput,self.buffer)
    return Convolution.accGradParameters(self,_input,_gradOutput,scale)
end

function TemporalConvolution:clearDesc()
  self.buffer = nil
  self._output = nil
  self.oSize = nil
end

function TemporalConvolution:write(f)
  self:clearDesc()
  cudnn.SpatialConvolution.clearDesc(self)
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function TemporalConvolution:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_input', '_gradOutput')
   return parent.clearState(self)
end
