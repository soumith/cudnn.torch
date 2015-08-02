require 'cudnn'

m = cudnn.SpatialConvolution(512,512,13,13,1,1,1,1,512)


inp = torch.zeros(1,512,512,512)

inp = inp:cuda()
m = m:cuda()

cutorch.reserveStreams(10)
-- cutorch.setStream(2) -- disables groups parallelization

local tm = os.clock()
for i=1,10 do
   o=m:forward(inp)
   cutorch.synchronize()
   print(os.clock() - tm)
   tm = os.clock()
end

print(#o)
