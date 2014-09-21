require 'cudnn'

i = torch.randn(128, 3, 100, 100):cuda()

m1 = nn.Sequential()
m1:add(cudnn.SpatialConvolution(3,16,5,5))
m1:add(cudnn.SpatialMaxPooling(2,2,2,2))
m1:cuda()
m2 = m1:clone():cuda()


o1=m1:forward(i)
o2=m2:forward(i)
