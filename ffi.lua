local ffi = require 'ffi'

ffi.cdef[[
size_t cudnnGetVersion();
struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;
typedef enum
{
    CUDNN_STATUS_SUCCESS          = 0,
    CUDNN_STATUS_NOT_INITIALIZED  = 1,
    CUDNN_STATUS_ALLOC_FAILED     = 2,
    CUDNN_STATUS_BAD_PARAM        = 3,
    CUDNN_STATUS_INTERNAL_ERROR   = 4,
    CUDNN_STATUS_INVALID_VALUE    = 5,
    CUDNN_STATUS_ARCH_MISMATCH    = 6,
    CUDNN_STATUS_MAPPING_ERROR    = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED    = 9,
    CUDNN_STATUS_LICENSE_ERROR    = 10
} cudnnStatus_t;

const char * cudnnGetErrorString(cudnnStatus_t status);

typedef struct CUstream_st *cudaStream_t;
cudnnStatus_t  cudnnCreate(cudnnHandle_t *handle);
cudnnStatus_t  cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId);

typedef struct cudnnTensorStruct*        cudnnTensorDescriptor_t;
typedef struct cudnnConvolutionStruct*   cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct*       cudnnPoolingDescriptor_t;
typedef struct cudnnFilterStruct*        cudnnFilterDescriptor_t;
typedef struct cudnnLRNStruct*           cudnnLRNDescriptor_t;

typedef enum
{
    CUDNN_DATA_FLOAT  = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF   = 2,
} cudnnDataType_t;

typedef enum
{
    CUDNN_TENSOR_NCHW = 0,   /* row major (wStride = 1, hStride = w) */
    CUDNN_TENSOR_NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
} cudnnTensorFormat_t;

cudnnStatus_t cudnnCreateTensorDescriptor( cudnnTensorDescriptor_t *tensorDesc);
cudnnStatus_t cudnnSetTensorNdDescriptor(  cudnnTensorDescriptor_t tensorDesc,
                                                       cudnnDataType_t dataType,
                                                       int nbDims,
                                                       const int dimA[],
                                                       const int strideA[]
                                                     );
cudnnStatus_t cudnnDestroyTensorDescriptor( cudnnTensorDescriptor_t tensorDesc);

typedef enum
{
   CUDNN_ADD_IMAGE   = 0,
   CUDNN_ADD_SAME_HW = 0,
   CUDNN_ADD_FEATURE_MAP = 1,
   CUDNN_ADD_SAME_CHW    = 1,
   CUDNN_ADD_SAME_C      = 2,
   CUDNN_ADD_FULL_TENSOR = 3
} cudnnAddMode_t;

cudnnStatus_t cudnnAddTensor_v2(cudnnHandle_t                    handle,
                             cudnnAddMode_t                   mode,
                             const void                      *alpha,
                             const cudnnTensorDescriptor_t    biasDesc,
                             const void                      *biasData,
                             const void                      *beta,
                             cudnnTensorDescriptor_t          srcDestDesc,
                             void                            *srcDestData
                             );

cudnnStatus_t cudnnSetTensor( cudnnHandle_t                   handle,
                              const cudnnTensorDescriptor_t   srcDestDesc,
                              void                           *srcDestData,
                              const void                     *value
                                         );

cudnnStatus_t cudnnScaleTensor(cudnnHandle_t                    handle,
                               const cudnnTensorDescriptor_t    srcDestDesc,
                               void                            *srcDestData,
                               const void                      *alpha
                               );

typedef enum
{
    CUDNN_CONVOLUTION       = 0,
    CUDNN_CROSS_CORRELATION = 1
} cudnnConvolutionMode_t;

typedef enum
{
    CUDNN_CONVOLUTION_FWD         = 0,        /* Tensor Convolution function */
    CUDNN_CONVOLUTION_WEIGHT_GRAD = 1,        /* Weight Gradient update function */
    CUDNN_CONVOLUTION_DATA_GRAD   = 2         /* Data Gradient update function */
} cudnnConvolutionPath_t;

cudnnStatus_t  cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc);

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t dataType,
                                         int nbDims,
                                         const int filterDimA[]
                                         );

cudnnStatus_t cudnnDestroyFilterDescriptor( cudnnFilterDescriptor_t filterDesc);

cudnnStatus_t
   cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc );

cudnnStatus_t
cudnnSetConvolutionNdDescriptor_v3( cudnnConvolutionDescriptor_t convDesc,
            int arrayLength,
            const int padA[],
            const int filterStrideA[],
            const int upscaleA[],
            cudnnConvolutionMode_t mode,
            cudnnDataType_t dataType
            );

cudnnStatus_t
   cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                   int arrayLength, /* nbDims-2 size */
                                   const int padA[],
                                   const int filterStrideA[],
                                   const int upscaleA[],
                                   cudnnConvolutionMode_t mode
                                   );

cudnnStatus_t
   cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                              int arrayLengthRequested,
                                              int *arrayLength,
                                              int padA[],
                                              int strideA[],
                                              int upscaleA[],
                                              cudnnConvolutionMode_t *mode
                                              );
cudnnStatus_t
   cudnnGetConvolutionNdForwardOutputDim(
           const cudnnConvolutionDescriptor_t convDesc,
           const cudnnTensorDescriptor_t inputTensorDesc,
           const cudnnFilterDescriptor_t filterDesc,
           int nbDims,
           int tensorOuputDimA[]
           );

/* Destroy an instance of convolution descriptor */
cudnnStatus_t cudnnDestroyConvolutionDescriptor(
  cudnnConvolutionDescriptor_t convDesc );

typedef enum
{
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE        = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST      = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2
} cudnnConvolutionFwdPreference_t;

typedef enum
{
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4
} cudnnConvolutionFwdAlgo_t;

typedef struct {
    cudnnConvolutionFwdAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionFwdAlgoPerf_t;

cudnnStatus_t
cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t                      handle,
             const cudnnTensorDescriptor_t      srcDesc,
             const cudnnFilterDescriptor_t      filterDesc,
             const cudnnConvolutionDescriptor_t convDesc,
             const cudnnTensorDescriptor_t      destDesc,
             const int                          requestedCount,
             int                                *returnedCount,
             cudnnConvolutionFwdAlgoPerf_t      *perfResults
             );


cudnnStatus_t cudnnGetConvolutionForwardAlgorithm( cudnnHandle_t handle,
                              const cudnnTensorDescriptor_t      srcDesc,
                              const cudnnFilterDescriptor_t      filterDesc,
                              const cudnnConvolutionDescriptor_t convDesc,
                              const cudnnTensorDescriptor_t      destDesc,
                              cudnnConvolutionFwdPreference_t    preference,
                              size_t                      memoryLimitInbytes,
                              cudnnConvolutionFwdAlgo_t         *algo
                                         );

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize( cudnnHandle_t handle,
                                 const cudnnTensorDescriptor_t      srcDesc,
                                 const cudnnFilterDescriptor_t      filterDesc,
                                 const cudnnConvolutionDescriptor_t convDesc,
                                 const cudnnTensorDescriptor_t      destDesc,
                                 cudnnConvolutionFwdAlgo_t          algo,
                                 size_t                            *sizeInBytes
                                 );


/* Function to perform the forward multiconvolution */
cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t                 handle,
                              const void                         *alpha,
                              const cudnnTensorDescriptor_t       srcDesc,
                              const void                         *srcData,
                              const cudnnFilterDescriptor_t       filterDesc,
                              const void                         *filterData,
                              const cudnnConvolutionDescriptor_t  convDesc,
                              cudnnConvolutionFwdAlgo_t           algo,
                              void                               *workSpace,
                              size_t                    workSpaceSizeInBytes,
                              const void                         *beta,
                              const cudnnTensorDescriptor_t       destDesc,
                              void                               *destData
                                                 );

/* Functions to perform the backward multiconvolution */
cudnnStatus_t cudnnConvolutionBackwardBias(   cudnnHandle_t           handle,
                                       const void                     *alpha,
                                       const cudnnTensorDescriptor_t   srcDesc,
                                       const void                      *srcData,
                                       const void                      *beta,
                                       const cudnnTensorDescriptor_t   destDesc,
                                       void                           *destData
                                                      );

typedef enum
{
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE        = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST      = 1
} cudnnConvolutionBwdFilterPreference_t;

typedef enum
{
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0         = 0,  // non-deterministic
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1         = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT       = 2
} cudnnConvolutionBwdFilterAlgo_t;

typedef struct {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionBwdFilterAlgoPerf_t;

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm( cudnnHandle_t handle,
             const cudnnTensorDescriptor_t          srcDesc,
             const cudnnTensorDescriptor_t          diffDesc,
             const cudnnConvolutionDescriptor_t     convDesc,
             const cudnnFilterDescriptor_t          gradDesc,
             const int                              requestedAlgoCount,
             int                                   *returnedAlgoCount,
             cudnnConvolutionBwdFilterAlgoPerf_t   *perfResults
                                                                     );


cudnnStatus_t
cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t          srcDesc,
    const cudnnTensorDescriptor_t          diffDesc,
    const cudnnConvolutionDescriptor_t     convDesc,
    const cudnnFilterDescriptor_t          gradDesc,
    cudnnConvolutionBwdFilterPreference_t  preference,
    size_t                                 memoryLimitInbytes,
    cudnnConvolutionBwdFilterAlgo_t        *algo
             );

cudnnStatus_t
cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnnHandle_t handle,
      const cudnnTensorDescriptor_t       srcDesc,
      const cudnnTensorDescriptor_t       diffDesc,
      const cudnnConvolutionDescriptor_t  convDesc,
      const cudnnFilterDescriptor_t       gradDesc,
      cudnnConvolutionBwdFilterAlgo_t     algo,
      size_t                              *sizeInBytes
                 );

cudnnStatus_t cudnnConvolutionBackwardFilter_v3(
       cudnnHandle_t                       handle,
       const void                         *alpha,
       const cudnnTensorDescriptor_t       srcDesc,
       const void                         *srcData,
       const cudnnTensorDescriptor_t       diffDesc,
       const void                         *diffData,
       const cudnnConvolutionDescriptor_t  convDesc,
       cudnnConvolutionBwdFilterAlgo_t     algo,
       void                               *workSpace,
       size_t                              workSpaceSizeInBytes,
       const void                         *beta,
       const cudnnFilterDescriptor_t       gradDesc,
       void                               *gradData
            );

typedef enum
{
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE        = 0,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST      = 1
} cudnnConvolutionBwdDataPreference_t;

typedef enum
{
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0         = 0, // non-deterministic
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1         = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT       = 2,
} cudnnConvolutionBwdDataAlgo_t;

typedef struct {
    cudnnConvolutionBwdDataAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionBwdDataAlgoPerf_t;


cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
             const cudnnFilterDescriptor_t       filterDesc,
             const cudnnTensorDescriptor_t       diffDesc,
             const cudnnConvolutionDescriptor_t  convDesc,
             const cudnnTensorDescriptor_t       gradDesc,
             const int                           requestedAlgoCount,
             int                                *returnedAlgoCount,
             cudnnConvolutionBwdDataAlgoPerf_t  *perfResults
                                                                   );

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
       cudnnHandle_t                      handle,
       const cudnnFilterDescriptor_t      filterDesc,
       const cudnnTensorDescriptor_t       diffDesc,
       const cudnnConvolutionDescriptor_t convDesc,
       const cudnnTensorDescriptor_t       gradDesc,
       cudnnConvolutionBwdDataPreference_t    preference,
       size_t                             memoryLimitInbytes,
       cudnnConvolutionBwdDataAlgo_t         *algo
                   );

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
          cudnnHandle_t                      handle,
          const cudnnFilterDescriptor_t      filterDesc,
          const cudnnTensorDescriptor_t       diffDesc,
          const cudnnConvolutionDescriptor_t convDesc,
          const cudnnTensorDescriptor_t       gradDesc,
          cudnnConvolutionBwdDataAlgo_t          algo,
          size_t                            *sizeInBytes
                 );


cudnnStatus_t cudnnConvolutionBackwardData_v3(
         cudnnHandle_t                       handle,
         const void                         *alpha,
         const cudnnFilterDescriptor_t       filterDesc,
         const void                         *filterData,
         const cudnnTensorDescriptor_t       diffDesc,
         const void                         *diffData,
         const cudnnConvolutionDescriptor_t  convDesc,
         cudnnConvolutionBwdDataAlgo_t           algo,
         void                               *workSpace,
         size_t                              workSpaceSizeInBytes,
         const void                         *beta,
         const cudnnTensorDescriptor_t       gradDesc,
         void                               *gradData
                );


cudnnStatus_t cudnnConvolutionBackwardFilter( cudnnHandle_t            handle,
                      const void                         *alpha,
                      const cudnnTensorDescriptor_t       srcDesc,
                      const void                         *srcData,
                      const cudnnTensorDescriptor_t       diffDesc,
                      const void                         *diffData,
                      const cudnnConvolutionDescriptor_t  convDesc,
                      const void                         *beta,
                      const cudnnFilterDescriptor_t       gradDesc,
                      void                               *gradData
                                      );


cudnnStatus_t cudnnConvolutionBackwardData(  cudnnHandle_t handle,
        const void                         *alpha,
        const cudnnFilterDescriptor_t       filterDesc,
        const void                         *filterData,
                          const cudnnTensorDescriptor_t       diffDesc,
                          const void                         *diffData,
                          const cudnnConvolutionDescriptor_t  convDesc,
                          const void                         *beta,
                          const cudnnTensorDescriptor_t       gradDesc,
                          void                               *gradData
                        );


/*
 *  softmax algorithm
 */
typedef enum
{
    CUDNN_SOFTMAX_FAST     = 0,
    CUDNN_SOFTMAX_ACCURATE = 1,
    CUDNN_SOFTMAX_LOG      = 2
} cudnnSoftmaxAlgorithm_t;

typedef enum
{
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,
    CUDNN_SOFTMAX_MODE_CHANNEL = 1
} cudnnSoftmaxMode_t;

/* Function to perform forward softmax */
cudnnStatus_t cudnnSoftmaxForward(  cudnnHandle_t                    handle,
            cudnnSoftmaxAlgorithm_t          algorithm,
            cudnnSoftmaxMode_t               mode,
            const void                      *alpha,
            const cudnnTensorDescriptor_t    srcDesc,
            const void                      *srcData,
            const void                      *beta,
            const cudnnTensorDescriptor_t    destDesc,
            void                            *destData
            );

/* Function to perform backward softmax */
cudnnStatus_t cudnnSoftmaxBackward(
     cudnnHandle_t                    handle,
                 cudnnSoftmaxAlgorithm_t          algorithm,
                 cudnnSoftmaxMode_t               mode,
                 const void                      *alpha,
                 const cudnnTensorDescriptor_t    srcDesc,
                 const void                      *srcData,
                 const cudnnTensorDescriptor_t    srcDiffDesc,
                 const void                      *srcDiffData,
                 const void                      *beta,
                 const cudnnTensorDescriptor_t    destDiffDesc,
                 void                            *destDiffData
               );

typedef enum
{
    CUDNN_POOLING_MAX     = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
    CUDNN_POOLING_AVERAGE = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING // for backward compatibility
} cudnnPoolingMode_t;

/* Create an instance of pooling descriptor */
cudnnStatus_t cudnnCreatePoolingDescriptor(
              cudnnPoolingDescriptor_t *poolingDesc);
cudnnStatus_t cudnnSetPoolingNdDescriptor(
       cudnnPoolingDescriptor_t poolingDesc,
                         const cudnnPoolingMode_t mode,
                         int nbDims,
                         const int windowDimA[],
                         const int paddingA[],
                         const int strideA[]
            );

cudnnStatus_t cudnnGetPoolingNdDescriptor(
                         const cudnnPoolingDescriptor_t poolingDesc,
                         const int nbDimsRequested,
                         cudnnPoolingMode_t *mode,
                         int *nbDims,
                         int windowDimA[],
                         int paddingA[],
                         int strideA[]
                      );

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim(
   const cudnnPoolingDescriptor_t poolingDesc,
                              const cudnnTensorDescriptor_t inputTensorDesc,
                              int nbDims,
                              int outputTensorDimA[]);
/* Destroy an instance of pooling descriptor */
cudnnStatus_t cudnnDestroyPoolingDescriptor(
   cudnnPoolingDescriptor_t poolingDesc );

/* Function to perform forward pooling */
cudnnStatus_t cudnnPoolingForward(  cudnnHandle_t handle,
                 const cudnnPoolingDescriptor_t   poolingDesc,
                 const void                      *alpha,
                 const cudnnTensorDescriptor_t    srcDesc,
                 const void                      *srcData,
                 const void                      *beta,
                 const cudnnTensorDescriptor_t    destDesc,
                 void                            *destData
              );

/* Function to perform backward pooling */
cudnnStatus_t cudnnPoolingBackward( cudnnHandle_t                   handle,
                 const cudnnPoolingDescriptor_t  poolingDesc,
                 const void                      *alpha,
                 const cudnnTensorDescriptor_t   srcDesc,
                 const void                     *srcData,
                 const cudnnTensorDescriptor_t   srcDiffDesc,
                 const void                     *srcDiffData,
                 const cudnnTensorDescriptor_t   destDesc,
                 const void                     *destData,
                 const void                     *beta,
                 const cudnnTensorDescriptor_t   destDiffDesc,
                 void                           *destDiffData
                                              );

typedef enum
{
    CUDNN_ACTIVATION_SIGMOID = 0,
    CUDNN_ACTIVATION_RELU    = 1,
    CUDNN_ACTIVATION_TANH    = 2
} cudnnActivationMode_t;

/* Function to perform forward activation  */
cudnnStatus_t cudnnActivationForward( cudnnHandle_t                    handle,
                   cudnnActivationMode_t            mode,
                   const void                      *alpha,
                   const cudnnTensorDescriptor_t    srcDesc,
                   const void                      *srcData,
                   const void                      *beta,
                   const cudnnTensorDescriptor_t    destDesc,
                   void                            *destData
                 );

/* Function to perform backward activation  */
cudnnStatus_t cudnnActivationBackward( cudnnHandle_t                    handle,
                    cudnnActivationMode_t            mode,
                    const void                      *alpha,
                    const cudnnTensorDescriptor_t    srcDesc,
                    const void                      *srcData,
                    const cudnnTensorDescriptor_t    srcDiffDesc,
                    const void                      *srcDiffData,
                    const cudnnTensorDescriptor_t    destDesc,
                    const void                      *destData,
                    const void                      *beta,
                    const cudnnTensorDescriptor_t    destDiffDesc,
                    void                            *destDiffData
                  );

cudnnStatus_t cudnnCreateLRNDescriptor( cudnnLRNDescriptor_t* normDesc );

typedef enum
{
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,
    CUDNN_BATCHNORM_SPATIAL        = 1
} cudnnBatchNormMode_t;

// Derives a tensor descriptor from layer data descriptor for BatchNormalization scale, invVariance, bnBias, bnScale subtensors
// Use the tensor desc produced by these functions as the bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc parameters in
// Spatial and Per-activation Batch Normalization forward and backward functions.
// Note - derivedBnDesc has to be first created using cudnnCreateTensorDescriptor
// Note - dataDesc is the descriptor for the layer data and has to be setup with proper dimensions prior to calling these functions.
cudnnStatus_t cudnnDeriveBNTensorDescriptor(
                              cudnnTensorDescriptor_t derivedBnDesc,
                              const cudnnTensorDescriptor_t dataDesc,
                              cudnnBatchNormMode_t mode);

// This function performs a forward pass for Batch Normalization layer.
// In addition to resultTopData it accumulates the moving averages of the mean and inverse variances
cudnnStatus_t cudnnBatchNormalizationForwardTraining(
                              cudnnHandle_t                    handle,
                              cudnnBatchNormMode_t             mode,

                              const void                      *alpha, // alpha[0] = result blend factor
                              const void                      *beta, // beta[0] = dest layer blend factor

                              const cudnnTensorDescriptor_t    bottomDesc,
                              const void                      *bottomData, // NxCxHxW
                              void                            *resultTopData, // NxCxHxW

                              // Same shared desc for all the 6 tensors below in the argument list.
                              // Note that the data type for this descriptor has to be set as follows:
                              // type = (typeOf(bottomData) == half) ? float : typeof(bottomData)
                              // The dimensions for this tensor descriptor are dependent on the normalization mode
                              // For spatial normalization the tensors are expected to be 1D (of size C)
                              // (in this case normalization is performed across NxHxW)
                              // In per-activation mode the normalization is performed across N dimension only
                              // So the tensors are expected to have dimensions of CxHxW
                              const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,

                              // Note - bnScale is 'gamma' in paper's notation
                              const void                      *bnScaleData, // Mode-dependent dims
                              // Note - this bias parameter can effectively replace the bias in Conv and FCN layers
                              // (Which can be set to zero for efficiency)
                              // Note - bnBias is 'beta' in paper's notation
                              const void                      *bnBiasData, // Mode-dependent dims

                              // It is required that factor=1 is used for the very first call of a complete training cycle.
                              // This is necessary to properly initialize the moving average.
                              // Use a factor=1/(1+n) at N-th call to the function to get
                              // Cumulative Moving Average (CMA) behavior
                              // CMA[n] = (x[1]+...+x[n])/n
                              // Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
                              // ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
                              // CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1)
                              double                           exponentialAverageFactor,

                              // runningMean = newMean*factor + runningMean*(1-factor)
                              // if isTrainingPhase == false, these tensors will remain const
                              // and exponentialAverageFactor parameter is not used.

                              // Both of these pointers (running mean, inv variance) can be NULL but only at the same time.
                              void                            *resultRunningMean,
                              // The value stored here (or passed as an input in inference mode) is the moving average
                              // of the expression 1 / sqrt( epsilon + variance[bottomData] )
                              void                            *resultRunningInvVariance,

                              // Constant used to prevent divides by zero variance. Has to be >= CUDNN_BN_MIN_EPSILON.
                              // Same epsilon value should be used in forward and backward functions.
                              double                           epsilon,

                              // Optional cache to save intermediate results computed during the forward pass
                              // - these can then be reused to speed up backward pass. For this to work correctly,
                              // the bottom layer data has to remain unchanged until the backward function is called.
                              // Note that both of these parameters can be NULL but only at the same time.
                              // It is recommended to use this cache since memory overhead is relatively small.
                              void                            *resultSaveMean,
                              void                            *resultSaveInvVariance
                              );

// This function will compute a linear transform of the inputs as follows:
// topData[i] = bnScale[k]*(bottomData[i]-estimatedMean[k])*estimatedInvVariance[k] + bnBias[k]
// with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
// according to spatial or per-activation mode (please refer to the paper for details).
// During inference estimatedMean and estimatedVariance are treated
// as const inputs (accumulated and saved during the training phase)
cudnnStatus_t cudnnBatchNormalizationForwardInference(
                              cudnnHandle_t                    handle,
                              cudnnBatchNormMode_t             mode,

                              const void                      *alpha, // alpha[0] = result blend factor
                              const void                      *beta, // beta[0] = dest layer blend factor

                              const cudnnTensorDescriptor_t    bottomDesc,
                              const void                      *bottomData, // NxCxHxW
                              void                            *resultTopData, // NxCxHxW

                              // Same desc for all 4 tensors below
                              // Note that the data type for this descriptor has to be set as follows:
                              // type = (typeOf(bottomData) == half) ? float : typeof(bottomData)
                              // The dimensions for this tensor descriptor are dependent on the normalization mode
                              // For spatial normalization the tensors are expected to be 1D (of size C)
                              // (in this case normalization is performed across NxHxW)
                              // In per-activation mode the normalization is performed across N dimension only
                              // So the tensors are expected to have dimensions of CxHxW
                              const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,

                              // Note - bnScale is 'gamma' in paper's notation
                              const void                      *bnScaleData, // Mode-dependent dims
                              // Note - this bias parameter can effectively replace the bias in Conv and FCN layers
                              // (Which can be set to zero for efficiency)
                              // Note - bnBias is 'beta' in paper's notation
                              const void                      *bnBiasData, // Mode-dependent dims

                              // runningMean = newMean*factor + runningMean*(1-factor)
                              // if isTrainingPhase == false, these tensors will remain const
                              // and exponentialAverageFactor parameter is not used.

                              // An estimate of the batch mean, can be accumulated over multiple calls to
                              // batchNormalizationForwardTraining
                              const void                      *estimatedMean,
                              // An estimate of the expression 1 / sqrt( epsilon + variance[bottomData] ),
                              // Can also be accumulated over multiple calls to batchNormalizationForwardTraining.
                              const void                      *estimatedInvVariance,

                              // Constant used to prevent divides by zero variance. Has to be >= CUDNN_BN_MIN_EPSILON.
                              // Same epsilon value should be used in forward and backward functions.
                              double                           epsilon
                              );


// This function performs a backward pass for Batch Normalization layer.
// The results are
// 1. bottom layer data differential
// 2. bnScale differential
// 3. bnBias differential
cudnnStatus_t cudnnBatchNormalizationBackward(
                              cudnnHandle_t                    handle,
                              cudnnBatchNormMode_t             mode,

                              const void                      *alpha, // result blend factor = alpha[0]
                              const void                      *beta, // bottom blend factor = beta[0]

                              const cudnnTensorDescriptor_t    bottomDesc, // same desc for topDiff, bottomDiff
                              const void                      *bottomData, // NxCxHxW
                              const void                      *topDiff, // NxCxHxW
                              void                            *resultBottomDiff, // NxCxHxW

                              // this tensor desc is used for all the 4 tensors below
                              const cudnnTensorDescriptor_t    bnScaleBiasDiffDesc,
                              const void                      *bottomBnScale, // bottomBnBias doesn't affect backpropagation

                              // scale and bias diff are not backpropagated below this layer (dead-end computation DAG nodes)
                              void                            *resultBnScaleDiff, // mode-dependent dims
                              void                            *resultBnBiasDiff, // mode-dependent dims
                              // Constant used to prevent divides by zero variance. Has to be >= CUDNN_BN_MIN_EPSILON.
                              // Same epsilon value should be used in forward and backward functions.
                              double                           epsilon,

                              // Optional cache parameters containing saved intermediate results computed during the forward pass
                              // For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called.
                              // Note that both of these parameters can be NULL but only at the same time.
                              // It is recommended to use this cache since memory overhead is relatively small.
                              const void                      *savedMean,
                              const void                      *savedInvVariance
                              );

typedef enum
  {
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0,
  } cudnnLRNMode_t;

cudnnStatus_t cudnnSetLRNDescriptor(
            cudnnLRNDescriptor_t   normDesc,
            unsigned               lrnN,
            double                 lrnAlpha,
            double                 lrnBeta,
            double                 lrnK);

cudnnStatus_t cudnnGetLRNDescriptor(
            cudnnLRNDescriptor_t   normDesc,
            unsigned*              lrnN,
            double*                lrnAlpha,
            double*                lrnBeta,
            double*                lrnK);

cudnnStatus_t cudnnDestroyLRNDescriptor( cudnnLRNDescriptor_t lrnDesc );

cudnnStatus_t cudnnLRNCrossChannelForward(
        cudnnHandle_t                    handle,
        cudnnLRNDescriptor_t             normDesc,
        cudnnLRNMode_t                   lrnMode,
        const void*                      alpha,
        const cudnnTensorDescriptor_t    srcDesc,
        const void                      *srcData,
        const void                      *beta,
        const cudnnTensorDescriptor_t    destDesc,
        void                            *destData);

cudnnStatus_t cudnnLRNCrossChannelBackward(
       cudnnHandle_t                    handle,
       cudnnLRNDescriptor_t             normDesc,
       cudnnLRNMode_t                   lrnMode,
       const void*                      alpha,
       const cudnnTensorDescriptor_t    srcDesc,
       const void                      *srcData,
       const cudnnTensorDescriptor_t    srcDiffDesc,
       const void                      *srcDiffData,
       const cudnnTensorDescriptor_t    destDesc,
       const void                      *destData,
       const void                      *beta,
       const cudnnTensorDescriptor_t    destDiffDesc,
       void                            *destDiffData);

typedef enum
  {
    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0,
  } cudnnDivNormMode_t;

cudnnStatus_t cudnnDivisiveNormalizationForward(
      cudnnHandle_t                    handle,
      cudnnLRNDescriptor_t             normDesc,
      cudnnDivNormMode_t               mode,
      const void                      *alpha,
      const cudnnTensorDescriptor_t    srcDesc,
      const void                      *srcData,
      const void                      *srcMeansData,
      void                            *tempData,
      void                            *tempData2,
      const void                      *beta,
      const cudnnTensorDescriptor_t    destDesc,
      void                            *destData
            );

cudnnStatus_t cudnnDivisiveNormalizationBackward(
             cudnnHandle_t                    handle,
             cudnnLRNDescriptor_t             normDesc,
             cudnnDivNormMode_t               mode,
             const void                      *alpha,
             const cudnnTensorDescriptor_t    srcDesc,
             const void                      *srcData,
             const void                      *srcMeansData,
             const void                      *srcDiffData,
             void                            *tempData,
             void                            *tempData2,
             const void                      *betaData,
             const cudnnTensorDescriptor_t    destDataDesc,
             void                            *destDataDiff,
             void                            *destMeansDiff
             );

]]

local ok,err = pcall(function() cudnn.C = ffi.load('libcudnn') end)
if not ok then
   print(err)
   error([['libcudnn.so not found in library path.
Please install CuDNN from https://developer.nvidia.com/cuDNN
Then make sure all the files named as libcudnn.so* are placed in your library load path (for example /usr/local/lib , or manually add a path to LD_LIBRARY_PATH)
]])
end

cudnn.version = tonumber(cudnn.C.cudnnGetVersion())
if cudnn.version < 3000 then
  error('These bindings are for version 3000 or above, '
        .. 'while the loaded CuDNN is version: ' .. cudnn.version
           .. '  \nAre you using an older version of CuDNN?')
end
