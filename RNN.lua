local RNN, parent = torch.class('cudnn.RNN', 'nn.Module')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

function RNN:__init(inputSize, hiddenSize, numLayers, batchFirst)
   parent.__init(self)

   self.datatype = 'CUDNN_DATA_FLOAT'
   self.inputSize = inputSize
   self.hiddenSize = hiddenSize
   self.seqLength = 1
   self.miniBatch = 1
   self.numLayers = numLayers
   self.bidirectional = 'CUDNN_UNIDIRECTIONAL'
   self.numDirections = 1 -- set to 2 for bi-directional.
   self.inputMode = 'CUDNN_LINEAR_INPUT'
   self.mode = 'CUDNN_RNN_RELU'
   self.dropout = 0
   self.seed = 0x01234567
   self.batchFirst = batchFirst or false -- Set to true for batch x time x inputdim.

   self.gradInput = torch.CudaTensor()
   self.output = torch.CudaTensor()
   self.weight = torch.CudaTensor()
   self.gradWeight = torch.CudaTensor()
   self.reserve = torch.CudaTensor()
   self.hiddenOutput = torch.CudaTensor()
   self.cellOutput = torch.CudaTensor()
   self.gradHiddenInput = torch.CudaTensor()
   self.gradCellInput = torch.CudaTensor()

   self:training()
   self:reset()
end

function RNN:reset(stdv)
   stdv = stdv or 1.0 / math.sqrt(self.hiddenSize)

   self:resetDropoutDescriptor()
   self:resetRNNDescriptor()
   self:resetIODescriptors()

   local weightSize = torch.LongTensor(1)
   errcheck('cudnnGetRNNParamsSize',
            cudnn.getHandle(),
            self.rnnDesc[0],
            self.xDescs,
            weightSize:data())
   weightSize[1] = (weightSize[1] + 3) / 4 -- sizeof(float)
   self.weight:resize(weightSize[1])
   self.weight:uniform(-stdv, stdv)
   self.gradWeight:resizeAs(self.weight):zero()
end

function RNN:createDescriptors(count, descs_type, create_func, destroy_func)
   local ds = ffi.new(descs_type, count)
   for i = 0, count - 1 do
      errcheck(create_func, ds + i)
   end
   local function destroyDescriptors(ds)
      for i = 0, count - 1 do
         errcheck(destroy_func, ds[i])
      end
   end
   ffi.gc(ds, destroyDescriptors)
   return ds
end

function RNN:createDropoutDescriptors(count)
   return self:createDescriptors(count,
                            'cudnnDropoutDescriptor_t[?]',
                            'cudnnCreateDropoutDescriptor',
                            'cudnnDestroyDropoutDescriptor')
end

function RNN:createFilterDescriptors(count)
   return self:createDescriptors(count,
                            'cudnnFilterDescriptor_t[?]',
                            'cudnnCreateFilterDescriptor',
                            'cudnnDestroyFilterDescriptor')
end

function RNN:createRNNDescriptors(count)
   return self:createDescriptors(count,
                            'cudnnRNNDescriptor_t[?]',
                            'cudnnCreateRNNDescriptor',
                            'cudnnDestroyRNNDescriptor')
end

function RNN:createTensorDescriptors(count)
   return self:createDescriptors(count,
                            'cudnnTensorDescriptor_t[?]',
                            'cudnnCreateTensorDescriptor',
                            'cudnnDestroyTensorDescriptor')
end

function RNN:resetDropoutDescriptor()
   if not self.dropoutDesc then
      self.dropoutDesc = self:createDropoutDescriptors(1)
   end

   self.dropoutStatesSize = torch.LongTensor(1)
   errcheck('cudnnDropoutGetStatesSize',
            cudnn.getHandle(),
            self.dropoutStatesSize:data())
            self.dropoutStates = torch.CudaTensor(self.dropoutStatesSize[1])

   errcheck('cudnnSetDropoutDescriptor',
            self.dropoutDesc[0],
            cudnn.getHandle(),
            self.dropout,
            self.dropoutStates:data(), self.dropoutStatesSize[1],
            self.seed)
end

function RNN:resetRNNDescriptor()
   if not self.rnnDesc then
      self.rnnDesc = self:createRNNDescriptors(1)
   end

   errcheck('cudnnSetRNNDescriptor',
            self.rnnDesc[0],
            self.hiddenSize,
            self.seqLength,
            self.numLayers,
            self.dropoutDesc[0],
            self.inputMode,
            self.bidirectional,
            self.mode,
            self.datatype)
end

function RNN:resetWeightDescriptor()
   if not self.wDesc then
      self.wDesc = self:createFilterDescriptors(1)
   end

   local dim = torch.IntTensor({self.weight:size(1), 1, 1})

   errcheck('cudnnSetFilterNdDescriptor',
            self.wDesc[0],
            self.datatype,
            'CUDNN_TENSOR_NCHW',
            3,
            dim:data())
end

function RNN:resetIODescriptors()
   self.xDescs = self:createTensorDescriptors(self.seqLength)
   self.yDescs = self:createTensorDescriptors(self.seqLength)

   for i = 0, self.seqLength - 1 do
      local dim = torch.IntTensor({self.inputSize, self.miniBatch, self.seqLength})
      local stride = torch.IntTensor({1, dim[1], dim[1] * dim[2]})
      errcheck('cudnnSetTensorNdDescriptor',
               self.xDescs[i],
               self.datatype,
               3,
               dim:data(),
               stride:data())

      local dim = torch.IntTensor({self.hiddenSize * self.numDirections, self.miniBatch, self.seqLength})
      local stride = torch.IntTensor({1, dim[1], dim[1] * dim[2]})
      errcheck('cudnnSetTensorNdDescriptor',
               self.yDescs[i],
               self.datatype,
               3,
               dim:data(),
               stride:data())
   end
end

function RNN:resetHiddenDescriptors()
   self.hxDesc = self:createTensorDescriptors(1)
   self.hyDesc = self:createTensorDescriptors(1)

   local dim = torch.IntTensor({self.hiddenSize, self.miniBatch, self.numLayers})
   local stride = torch.IntTensor({1, dim[1], dim[1] * dim[2]})

   errcheck('cudnnSetTensorNdDescriptor',
            self.hxDesc[0],
            self.datatype,
            3,
            dim:data(),
            stride:data())
   errcheck('cudnnSetTensorNdDescriptor',
            self.hyDesc[0],
            self.datatype,
            3,
            dim:data(),
            stride:data())
end

function RNN:resetCellDescriptors()
   self.cxDesc = self:createTensorDescriptors(1)
   self.cyDesc = self:createTensorDescriptors(1)

   local dim = torch.IntTensor({self.hiddenSize, self.miniBatch, self.numLayers})
   local stride = torch.IntTensor({1, dim[1], dim[1] * dim[2]})

   errcheck('cudnnSetTensorNdDescriptor',
            self.cxDesc[0],
            self.datatype,
            3,
            dim:data(),
            stride:data())
   errcheck('cudnnSetTensorNdDescriptor',
            self.cyDesc[0],
            self.datatype,
            3,
            dim:data(),
            stride:data())
end

function RNN:makeContiguous(input, gradOutput)
   if input and not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function RNN:resizeOutput(tensor)
    return tensor:resize(self.seqLength, self.miniBatch, self.hiddenSize * self.numDirections)
end

function RNN:resizeHidden(tensor)
    return tensor:resize(self.numLayers * self.numDirections, self.miniBatch, self.hiddenSize)
end

function RNN:updateOutput(input)
    if (self.batchFirst) then
        input = input:transpose(1, 2)
    end
   assert(input:dim() == 3, 'input must have 3 dimensions: seqLength, miniBatch, inputSize')
   -- Decide which descriptors/tensors need to be updated.
   local resetRNN = not self.dropoutDesc or not self.rnnDesc
   local resetIO = not self.xDescs or not self.yDescs
   local resetHC = not self.hxDesc or not self.hyDesc or not self.cxDesc or not self.cyDesc
   local resetWeight = not self.wDesc

   if input:size(1) ~= self.seqLength then
      self.seqLength = input:size(1)
      resetRNN = true
      resetIO = true
   end

   if input:size(2) ~= self.miniBatch then
      self.miniBatch = input:size(2)
      resetIO = true
      resetHC = true
   end

   assert(input:size(3) == self.inputSize, 'Incorrect input size!')

   -- Update descriptors/tensors
   if resetRNN then
      self:resetDropoutDescriptor()
      self:resetRNNDescriptor()
   end
   if resetIO then
      self:resetIODescriptors(input)
   end
   if resetHC then
      self:resetHiddenDescriptors()
      self:resetCellDescriptors()
   end
   if resetWeight then
      self:resetWeightDescriptor()
   end

   local x = self:makeContiguous(input)
   local y = self:resizeOutput(self.output)
   local w = self.weight
   local hy = self:resizeHidden(self.hiddenOutput):zero()
   local cy = self:resizeHidden(self.cellOutput):zero()

   -- Optionally use hiddenInput/cellInput parameters
   local hx = self.hiddenInput
   local cx = self.cellInput

   if hx then
      assert(hx:dim() == 3, 'hiddenInput must have 3 dimensions: numLayers, miniBatch, hiddenSize')
      assert(hx:size(1) == self.numLayers * self.numDirections, 'hiddenInput has incorrect number of layers!')
      assert(hx:size(2) == self.miniBatch, 'hiddenInput has incorrect number of minibathes!')
      assert(hx:size(3) == self.hiddenSize, 'hiddenIinput has incorrect size!')
      assert(hx:isContiguous(), 'hiddenInput must be contiguous!') end

   if cx then
      assert(cx:dim() == 3, 'cellInput must have 3 dimensions: numLayers, miniBatch, hiddenSize')
      assert(cx:size(1) == self.numLayers * self.numDirections, 'cellInput has incorrect number of layers!')
      assert(cx:size(2) == self.miniBatch, 'cellInput has incorrect number of minibathes!')
      assert(cx:size(3) == self.hiddenSize, 'cellInput has incorrect size!')
      assert(cx:isContiguous(), 'cellInput must be contiguous!')
   end

   self.workspace = cudnn.getSharedWorkspace()
   local workspaceSize = torch.LongTensor(1)
   errcheck('cudnnGetRNNWorkspaceSize',
            cudnn.getHandle(),
            self.rnnDesc[0],
            self.xDescs,
            workspaceSize:data())
   workspaceSize[1] = (workspaceSize[1] + 3) / 4 -- sizeof(float)
   if self.workspace:size(1) < workspaceSize[1] then
      self.workspace:resize(workspaceSize[1])
   end

   if self.train then
      local reserveSize = torch.LongTensor(1)
      errcheck('cudnnGetRNNTrainingReserveSize',
               cudnn.getHandle(),
               self.rnnDesc[0],
               self.xDescs,
               reserveSize:data())
      reserveSize[1] = (reserveSize[1] + 3) / 4 -- sizeof(float)
      if self.reserve:dim() == 0 or
         self.reserve:size(1) < reserveSize[1] then
         self.reserve:resize(reserveSize[1])
      end

      errcheck('cudnnRNNForwardTraining',
               cudnn.getHandle(),
               self.rnnDesc[0],
               self.xDescs, x:data(),
               self.hxDesc[0], hx and hx:data() or nil,
               self.cxDesc[0], cx and cx:data() or nil,
               self.wDesc[0], w:data(),
               self.yDescs, y:data(),
               self.hyDesc[0], hy:data(),
               self.cyDesc[0], cy:data(),
               self.workspace:data(), self.workspace:size(1) * 4, -- sizeof(float)
               self.reserve:data(), self.reserve:size(1) * 4) -- sizeof(float)
   else
      errcheck('cudnnRNNForwardInference',
               cudnn.getHandle(),
               self.rnnDesc[0],
               self.xDescs, x:data(),
               self.hxDesc[0], hx and hx:data() or nil,
               self.cxDesc[0], cx and cx:data() or nil,
               self.wDesc[0], w:data(),
               self.yDescs, y:data(),
               self.hyDesc[0], hy:data(),
               self.cyDesc[0], cy:data(),
               self.workspace:data(), self.workspace:size(1) * 4) -- sizeof(float)
   end
    if (self.batchFirst) then
        self.output = self.output:transpose(1, 2)
    end
   return self.output
end

function RNN:updateGradInput(input, gradOutput)
    if (self.batchFirst) then
        input = input:transpose(1, 2)
        gradOutput = gradOutput:transpose(1, 2)
    end
   assert(input:dim() == 3, 'input should have 3 dimensions: seqLength, miniBatch, inputSize')
   assert(input:size(1) == self.seqLength, 'input has incorrect sequence length!')
   assert(input:size(2) == self.miniBatch, 'input has incorrect minibatch size!')
   assert(input:size(3) == self.inputSize, 'input has incorrect size!')
   assert(self.train, 'updateGradInput can only be called when training!')
   local expectedSize = torch.LongStorage {self.seqLength, self.miniBatch, self.hiddenSize * self.numDirections}
   assert(gradOutput:isSize(expectedSize), 'gradOutput has incorrect size!')
   local x, dy = self:makeContiguous(nil, gradOutput) -- No need to calculate x.
   local y = self.output
   local w = self.weight
   local dx = self.gradInput:resizeAs(input)
   local hx = self.hiddenInput
   local cx = self.cellInput
   local dhy = self.gradHiddenOutput
   local dcy = self.gradCellOutput
   local dhx = self:resizeHidden(self.gradHiddenInput):zero()
   local dcx = self:resizeHidden(self.gradCellInput):zero()


   if hx then
      assert(hx:dim() == 3, 'hiddenInput must have 3 dimensions: numLayers, miniBatch, hiddenSize')
      assert(hx:size(1) == self.numLayers * self.numDirections, 'hiddenInput has incorrect number of layers!')
      assert(hx:size(2) == self.miniBatch, 'hiddenInput has incorrect minibatch size!')
      assert(hx:size(3) == self.hiddenSize, 'hiddenInput has incorrect size!')
      assert(hx:isContiguous(), 'hiddenInput must be contiguous!')
   end

   if cx then
      assert(cx:dim() == 3, 'cellInput must have 3 dimensions: numLayers, miniBatch, hiddenSize')
      assert(cx:size(1) == self.numLayers * self.numDirections, 'cellInput has incorrect number of layers!')
      assert(cx:size(2) == self.miniBatch, 'cellInput has incorrect minibatch size!')
      assert(cx:size(3) == self.hiddenSize, 'cellInput has incorrect size!')
      assert(cx:isContiguous(), 'cellInput must be contiguous!')
   end

   if dhy then
      assert(dhy:dim() == 3, 'gradHiddenOutput must have 3 dimensions: ' ..
                             'numLayers, miniBatch, hiddenSize')
      assert(dhy:size(1) == self.numLayers * self.numDirections, 'gradHiddenOutput has incorrect number of layers!')
      assert(dhy:size(2) == self.miniBatch, 'gradHiddenOutput has incorrect minibatch size!')
      assert(dhy:size(3) == self.hiddenSize, 'gradHiddenOutput has incorrect size!')
      assert(dhy:isContiguous(), 'gradHiddenOutput must be contiguous!')
   end

   if dcy then
      assert(dcy:dim() == 3, 'gradCellOutput must have 3 dimensions: ' ..
                             'numLayers, miniBatch, hiddenSize')
      assert(dcy:size(1) == self.numLayers * self.numDirections, 'gradCellOutput has incorrect number of layers!')
      assert(dcy:size(2) == self.miniBatch, 'gradCellOutput has incorrect minibatch size!')
      assert(dcy:size(3) == self.hiddenSize, 'gradCellOutput has incorrect size!')
      assert(dcy:isContiguous(), 'gradCellOutput must be contiguous!')
   end

   errcheck('cudnnRNNBackwardData',
            cudnn.getHandle(),
            self.rnnDesc[0],
            self.yDescs, y:data(),
            self.yDescs, dy:data(),
            self.hyDesc[0], dhy and dhy:data() or nil,
            self.cyDesc[0], dcy and dcy:data() or nil,
            self.wDesc[0], w:data(),
            self.hxDesc[0], hx and hx:data() or nil,
            self.cxDesc[0], cx and cx:data() or nil,
            self.xDescs, dx:data(),
            self.hxDesc[0], dhx:data(),
            self.cxDesc[0], dcx:data(),
            self.workspace:data(), self.workspace:size(1) * 4, -- sizeof(float)
            self.reserve:data(), self.reserve:size(1) * 4) -- sizeof(float)
    if (self.batchFirst) then
        self.gradInput = self.gradInput:transpose(1, 2)
    end
   return self.gradInput
end

function RNN:accGradParameters(input, gradOutput, scale)
    if (self.batchFirst) then
        input = input:transpose(1, 2)
        gradOutput = gradOutput:transpose(1, 2)
    end
   scale = scale or 1
   if scale == 0 then return end

   assert(input:dim() == 3, 'input should have 3 dimensions: seqLength, miniBatch, inputSize')
   assert(input:size(1) == self.seqLength, 'input has incorrect sequence length!')
   assert(input:size(2) == self.miniBatch, 'input has incorrect minibatch size!')
   assert(input:size(3) == self.inputSize, 'input has incorrect size!')
   local expectedSize = torch.LongStorage {self.seqLength, self.miniBatch, self.hiddenSize * self.numDirections}
   assert(gradOutput:isSize(expectedSize), 'gradOutput has incorrect size!')
   assert(self.train, 'accGradParameters can only be called when training!')

   local x, dy = self:makeContiguous(input, gradOutput)
   local hx = self.hiddenInput
   local y = self.output
   local dw = self.gradWeight

   if hx then
      assert(hx:dim() == 3, 'hiddenInput must have 3 dimensions: numLayers, miniBatch, hiddenSize')
      assert(hx:size(1) == self.numLayers * self.numDirections, 'hiddenInput has incorrect number of layers!')
      assert(hx:size(2) == self.miniBatch, 'hiddenInput has incorrect minibatch size!')
      assert(hx:size(3) == self.hiddenSize, 'hiddenIinput has incorrect size!')
      assert(hx:isContiguous(), 'hiddenInput must be contiguous!')
   end

   -- cudnnRNNBackwardWeights doesn't accept a scale parameter so instead
   -- scale before and after.
   -- TODO: How much does this impact accuracy?
   --       Use a secondary buffer instead?
   if scale ~= 1 then
      local scaleTensor = torch.Tensor({1 / scale})
      errcheck('cudnnScaleTensor',
               cudnn.getHandle(),
               self.wDesc[0],
               self.dw:data(),
               scaleTensor:data())
   end

   errcheck('cudnnRNNBackwardWeights',
            cudnn.getHandle(),
            self.rnnDesc[0],
            self.xDescs, x:data(),
            self.hxDesc[0], hx and hx:data() or nil,
            self.yDescs, y:data(),
            self.workspace:data(), self.workspace:size(1) * 4, -- sizeof(float)
            self.wDesc[0], dw:data(),
            self.reserve:data(), self.reserve:size(1) * 4) -- sizeof(float)

   if scale ~= 1 then
      local scaleTensor = torch.Tensor({scale})
      errcheck('cudnnScaleTensor',
               cudnn.getHandle(),
               self.wDesc[0],
               self.dw:data(),
               scaleTensor:data())
   end
end

function RNN:clearDesc()
   self.dropoutDesc = nil
   self.rnnDesc = nil
   self.dropoutDesc = nil
   self.wDesc = nil
   self.xDescs = nil
   self.yDescs = nil
   self.hxDesc = nil
   self.hyDesc = nil
   self.cxDesc = nil
   self.cyDesc = nil
end

function RNN:write(f)
   self:clearDesc()
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

function RNN:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_input', '_gradOutput', 'reserve', 'dropoutStates')
   return parent.clearState(self)
end
