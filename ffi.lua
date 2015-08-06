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

cudnnStatus_t cudnnAddTensor(cudnnHandle_t                    handle,
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
   cudnnGetConvolutionNdForwardOutputDim(
           const cudnnConvolutionDescriptor_t convDesc,
           const cudnnTensorDescriptor_t inputTensorDesc,
           const cudnnFilterDescriptor_t filterDesc,
           int nbDims,
           int tensorOuputDimA[]
           );

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
                       size_t                             memoryLimitInbytes,
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

cudnnStatus_t cudnnDestroyPoolingDescriptor(
		      cudnnPoolingDescriptor_t poolingDesc );

cudnnStatus_t cudnnPoolingForward(  cudnnHandle_t handle,
                 const cudnnPoolingDescriptor_t   poolingDesc,
                 const void                      *alpha,
                 const cudnnTensorDescriptor_t    srcDesc,
                 const void                      *srcData,
                 const void                      *beta,
                 const cudnnTensorDescriptor_t    destDesc,
                 void                            *destData
              );

cudnnStatus_t cudnnPoolingBackward( cudnnHandle_t handle,
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

cudnnStatus_t cudnnActivationForward( cudnnHandle_t handle,
                   cudnnActivationMode_t            mode,
                   const void                      *alpha,
                   const cudnnTensorDescriptor_t    srcDesc,
                   const void                      *srcData,
                   const void                      *beta,
                   const cudnnTensorDescriptor_t    destDesc,
                   void                            *destData
                 );

cudnnStatus_t cudnnActivationBackward( cudnnHandle_t handle,
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
