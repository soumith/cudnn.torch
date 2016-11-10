local VolumetricConvolution, parent
   = torch.class('cudnn.VolumetricConvolution', 'nn.VolumetricConvolution')
local ffi = require 'ffi'
local find = require 'cudnn.find'
local errcheck = cudnn.errcheck

local Convolution = cudnn.SpatialConvolution

-- if you change the configuration of the module manually, call this
function VolumetricConvolution:resetWeightDescriptors()
   local desc = {self.nOutputPlane, self.nInputPlane,
                 self.kT, self.kH, self.kW}
   return Convolution.resetWeightDescriptors(self,desc)
end

function VolumetricConvolution:fastest(mode)
   return Convolution.fastest(self,mode)
end

function VolumetricConvolution:setMode(fmode, bdmode, bwmode)
   return Convolution.setMode(self, fmode, bdmode, bwmode)
end

function VolumetricConvolution:resetMode()
   return Convolution.resetMode(self)
end

function VolumetricConvolution:createIODescriptors(input)
   if input:dim() == 4 then
      input = input:view(1, input:size(1), input:size(2),
                         input:size(3), input:size(4))
      batch = false
   end
   if Convolution.checkInputChanged(self, input) then
         -- create input descriptor
         self.iDesc = cudnn.toDescriptor(input)
         -- create conv descriptor
         self.pad = {self.padT, self.padH, self.padW}
         self.stride = {self.dT, self.dH, self.dW}

         local mathtype=cudnn.configmap(torch.type(self.weight))
         -- 3D convolutions do not work in 16 bits
         if mathtype == 'CUDNN_DATA_HALF' then
            mathtype = 'CUDNN_DATA_FLOAT'
         end
         self.convDescData = { padA = self.pad, filterStrideA = self.stride,
                               dataType = mathtype }
         self.convDesc = cudnn.setConvolutionDescriptor(self.convDescData)

         local oSize = torch.IntTensor(5)
         errcheck('cudnnGetConvolutionNdForwardOutputDim',
                  self.convDesc[0], self.iDesc[0],
                  self.weightDesc[0], 5, oSize:data())
         self.output:resize(oSize:long():storage())
         -- create descriptor for output
         self.oDesc = cudnn.toDescriptor(self.output)
         self.oDescForBias = cudnn.toDescriptor(
            self.output:view(self.output:size(1),
                             self.output:size(2),
                             self.output:size(3)*self.output:size(4),
                             self.output:size(5)))
         self.input_offset = 0
         self.output_offset = 0
         self.weight_offset = 0
-- next two lines are so that input does not get wiped out in clearState
-- otherwise, tests do not pass
         local input_slice = input:narrow(2,1,self.nInputPlane)
         local output_slice = self.output:narrow(2,1,self.nOutputPlane)
         find:prepare(self, input_slice, output_slice)
   end
end

function VolumetricConvolution:updateOutput(input)
   return Convolution.updateOutput(self, input)
end

function VolumetricConvolution:updateGradInput(input, gradOutput)
   return Convolution.updateGradInput(self, input, gradOutput)
end

function VolumetricConvolution:accGradParameters(input, gradOutput, scale)
   return Convolution.accGradParameters(self, input, gradOutput, scale)
end

function VolumetricConvolution:clearDesc()
   return Convolution.clearDesc(self)
end

function VolumetricConvolution:write(f)
   return Convolution.write(self, f)
end

function VolumetricConvolution:clearState()
   return Convolution.clearState(self)
end

return VolumetricConvolution
