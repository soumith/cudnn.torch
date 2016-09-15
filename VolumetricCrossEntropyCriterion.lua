local VolumetricCrossEntropyCriterion, parent = torch.class('cudnn.VolumetricCrossEntropyCriterion', 'nn.Criterion')

--[[
    This criterion does the VolumetricCrossEntropyCriterion across
    the feature dimension for a N-channel 3D image/video of TxHxW in size.

    It only supports mini-batches (5D input, 4D target)

    It does a LogSoftMax on the input (over the channel dimension),
    so no LogSoftMax is needed in the network at the end

    input = batchSize x nClasses x T x H x W
    target = batchSize x T x H x W
]]--

function VolumetricCrossEntropyCriterion:__init(weights)
    parent.__init(self)
    self.scec = cudnn.SpatialCrossEntropyCriterion(weights)
end

local foldInput = function(input)
    -- Fold time and height into one dimension
    -- bdthw -> bd(t*h)w
    input = input:view(input:size(1), input:size(2), 
                       input:size(3)*input:size(4), input:size(5))
    return input
end

local foldTarget = function(target)
    -- Fold time and height into one dimension
    -- bthw -> b(t*h)w
    target = target:view(target:size(1), target:size(2)*target:size(3), 
                         target:size(4))
    return target
end

function VolumetricCrossEntropyCriterion:updateOutput(input, target)
    assert(input:dim() == 5, 'mini-batch supported only')
    assert(target:dim() == 4, 'mini-batch supported only')
    assert(input:size(1) == target:size(1), 'input and target should be of same size')
    assert(input:size(3) == target:size(2), 'input and target should be of same size')
    assert(input:size(4) == target:size(3), 'input and target should be of same size')
    assert(input:size(5) == target:size(4), 'input and target should be of same size')

    -- Fold inputs and use spatial cross entropy criterion
    self.scec:updateOutput(foldInput(input), foldTarget(target))
    self.output = self.scec.output
    return self.output
end

function VolumetricCrossEntropyCriterion:updateGradInput(input, target)
    assert(input:dim() == 5, 'mini-batch supported only')
    assert(target:dim() == 4, 'mini-batch supported only')
    assert(input:size(1) == target:size(1), 'input and target should be of same size')
    assert(input:size(3) == target:size(2), 'input and target should be of same size')
    assert(input:size(4) == target:size(3), 'input and target should be of same size')
    assert(input:size(5) == target:size(4), 'input and target should be of same size')

    local originalInputSize = input:size()
    self.scec:updateGradInput(foldInput(input), foldTarget(target))
    self.gradInput = self.scec.gradInput:view(originalInputSize)
    return self.gradInput
end