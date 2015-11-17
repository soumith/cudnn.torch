require 'nn'

local SpatialCrossEntropyCriterion, parent = torch.class('cudnn.SpatialCrossEntropyCriterion', 'nn.Criterion')

--[[
    This criterion does the SpatialCrossEntropyCriterion across
    the feature dimension for a N-channel image of HxW in size.

    It only supports mini-batches (4D input, 3D target)

    It does a LogSoftMax on the input (over the channel dimension),
    so no LogSoftMax is needed in the network at the end

    input = batchSize x nClasses x H x W
    target = batchSize x H x W
]]--
function SpatialCrossEntropyCriterion:__init(weights)
    parent.__init(self)
    self.slsm = cudnn.SpatialLogSoftMax()
    self.nll = nn.ClassNLLCriterion(weights)
    self.sizeAverage = true
end

local transpose = function(input)
    input = input:transpose(2,4):transpose(2,3):contiguous() -- bdhw -> bwhd -> bhwd
    input = input:view(input:size(1)*input:size(2)*input:size(3), input:size(4))
    return input
end

local transposeBack = function(input, originalInput)
    input = input:view(originalInput:size(1), originalInput:size(3),
                       originalInput:size(4), originalInput:size(2))
    input = input:transpose(2,4):transpose(3,4):contiguous()  -- bhwd -> bdwh -> bdhw
    return input
end

function SpatialCrossEntropyCriterion:updateOutput(input, target)
    assert(input:dim() == 4, 'mini-batch supported only')
    assert(target:dim() == 3, 'mini-batch supported only')
    assert(input:size(1) == target:size(1), 'input and target should be of same size')
    assert(input:size(3) == target:size(2), 'input and target should be of same size')
    assert(input:size(4) == target:size(3), 'input and target should be of same size')
    -- apply SpatialLogSoftMax to input
    self.slsm:updateOutput(input)

    -- Update submodule sizeAverage to make it consistent.
    self.nll.sizeAverage = self.sizeAverage

    -- fold the height and width dims into the mini-batch dim.
    self.nll:updateOutput(transpose(self.slsm.output), target:view(-1))
    self.output = self.nll.output
    return self.output
end

function SpatialCrossEntropyCriterion:updateGradInput(input, target)
    assert(input:dim() == 4, 'mini-batch supported only')
    assert(target:dim() == 3, 'mini-batch supported only')
    assert(input:size(1) == target:size(1), 'input and target should be of same size')
    assert(input:size(3) == target:size(2), 'input and target should be of same size')
    assert(input:size(4) == target:size(3), 'input and target should be of same size')

    self.nll:updateGradInput(transpose(self.slsm.output), target:view(-1))

    -- unfold the height and width dims back
    self.slsm:updateGradInput(input, transposeBack(self.nll.gradInput, input))
    self.gradInput = self.slsm.gradInput
    return self.gradInput
end

function SpatialCrossEntropyCriterion:type(type)
    if type then
        self.nll:type(type)
        self.slsm:type(type)
    end
    parent.type(self, type)
    return self
end
