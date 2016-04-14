local VolumetricBatchNormalization =
   torch.class('cudnn.VolumetricBatchNormalization', 'cudnn.BatchNormalization')

VolumetricBatchNormalization.mode = 'CUDNN_BATCHNORM_SPATIAL'
VolumetricBatchNormalization.nDim = 5
VolumetricBatchNormalization.__version = 2
