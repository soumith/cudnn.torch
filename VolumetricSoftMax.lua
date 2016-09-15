local VolumetricSoftMax, parent = torch.class('cudnn.VolumetricSoftMax', 'nn.Module')

function VolumetricSoftMax:__init(fast)
    parent.__init(self)
    self.ssm = cudnn.SpatialSoftMax(fast)
end

local fold = function(input)
    -- Fold time and height into one dimension
    if input:dim() == 4 then
        -- dthw -> d(t*h)w
        input = input:view(input:size(1), input:size(2)*input:size(3),
                           input:size(4))
    else
        -- bdthw -> bd(t*h)w
        input = input:view(input:size(1), input:size(2), 
                           input:size(3)*input:size(4), input:size(5))
    end
    return input
end

function VolumetricSoftMax:updateOutput(input)
    assert(input:dim() == 4 or input:dim() == 5, 
           'input should either be a 3d image or a minibatch of them')
    local originalInputSize = input:size()

    -- Apply SpatialSoftMax to folded input
    self.ssm:updateOutput(fold(input))
    self.output = self.ssm.output:view(originalInputSize)
    return self.output
end

function VolumetricSoftMax:updateGradInput(input, gradOutput)
    assert(input:dim() == 4 or input:dim() == 5, 
           'input should either be a 3d image or a minibatch of them')

    local originalInputSize = input:size()
    self.ssm:updateGradInput(fold(input), fold(gradOutput))

    self.gradInput = self.ssm.gradInput:view(originalInputSize)
    return self.gradInput
end

function VolumetricSoftMax:clearState()
   self.ssm:clearState()
   return parent.clearState(self)
end