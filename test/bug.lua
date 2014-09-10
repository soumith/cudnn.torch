require 'cutorch'
require 'cudnn'

local gconv = cudnn.SpatialConvolution(3,16,5,5):cuda()
local inp = torch.randn(128, 3, 100, 100):cuda()
local out = gconv:forward(inp)
-- runs ok
for i=1,100 do
   out = gconv:forward(inp)
end
print('ok!')

print('problemo!')
inp = torch.randn(128, 3, 200, 200):cuda() -- change input shape
out = gconv:forward(inp)
