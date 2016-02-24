local SpatialBatchNormalization, parent =
   torch.class('cudnn.SpatialBatchNormalization', 'cudnn.BatchNormalization')

SpatialBatchNormalization.mode = 'CUDNN_BATCHNORM_SPATIAL'
SpatialBatchNormalization.nDim = 4
