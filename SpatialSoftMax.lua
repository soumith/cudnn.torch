local SpatialSoftMax, parent = torch.class('cudnn.SpatialSoftMax', 'nn.Module')
local errcheck = cudnn.errcheck

function SpatialSoftMax:__init(fast)
   parent.__init(self)
   if fast then
      self.algorithm = 'CUDNN_SOFTMAX_FAST'
   else
      self.algorithm = 'CUDNN_SOFTMAX_ACCURATE'
   end
end

function SpatialSoftMax:createIODescriptors(input)
   self.mode = self.mode or 'CUDNN_SOFTMAX_MODE_CHANNEL'
   -- after converting from nn use accurate
   self.algorithm = self.algorithm or 'CUDNN_SOFTMAX_ACCURATE'
   self.iSize = self.iSize or torch.LongStorage(4):fill(0)

   local batch = true
   local singleDim = false
   if input:dim() == 1 then
      singleDim = true
      batch = false
      input = input:view(1, input:size(1), 1, 1)
   elseif input:dim() == 2 then
      singleDim = true
      input = input:view(input:size(1), input:size(2), 1, 1)
   elseif input:dim() == 3 then
      input = input:view(1, input:size(1), input:size(2), input:size(3))
      batch = false
   end
   assert(input:dim() == 4 and input:isContiguous());

   if not self.iDesc or not self.oDesc or
      input:size(1) ~= self.iSize[1] or input:size(2) ~= self.iSize[2]
   or input:size(3) ~= self.iSize[3] or input:size(4) ~= self.iSize[4] then
      self.iSize = input:size()
      self.output:resizeAs(input)
      self.iDesc = cudnn.toDescriptor(input)
      self.oDesc = cudnn.toDescriptor(self.output)
      if not singleDim and not batch then
         self.output = self.output:view(self.output:size(2),
                                        self.output:size(3),
                                        self.output:size(4))
      elseif singleDim and not batch then
         self.output = self.output:view(self.output:size(2))
      elseif singleDim and batch then
         self.output = self.output:view(self.output:size(1), self.output:size(2))
      end
   end
end




function SpatialSoftMax:updateOutput(input)
   self:createIODescriptors(input)
   errcheck('cudnnSoftmaxForward',
            cudnn.getHandle(),
            self.algorithm, self.mode,
            cudnn.scalar(input, 1),
            self.iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            self.oDesc[0], self.output:data());
   return self.output
end

function SpatialSoftMax:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end

   self:createIODescriptors(input)
   errcheck('cudnnSoftmaxBackward',
            cudnn.getHandle(),
            self.algorithm, self.mode,
            cudnn.scalar(input, 1),
            self.oDesc[0], self.output:data(),
            self.oDesc[0], gradOutput:data(),
            cudnn.scalar(input, 0),
            self.iDesc[0], self.gradInput:data());
   return self.gradInput
end

function SpatialSoftMax:clearDesc()
   self.iDesc = nil
   self.oDesc = nil
end

function SpatialSoftMax:write(f)
   self:clearDesc()
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function SpatialSoftMax:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_gradOutput')
   return parent.clearState(self)
end
