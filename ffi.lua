local ffi = require 'ffi'

ffi.cdef[[
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

typedef struct CUstream_st *cudaStream_t;
cudnnStatus_t  cudnnCreate(cudnnHandle_t *handle);
cudnnStatus_t  cudnnDestroy(cudnnHandle_t handle);
typedef struct cudnnTensor4dStruct*      cudnnTensor4dDescriptor_t;
typedef struct cudnnConvolutionStruct*   cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct*       cudnnPoolingDescriptor_t;
typedef struct cudnnFilterStruct*        cudnnFilterDescriptor_t;
typedef enum
{
    CUDNN_DATA_FLOAT  = 0,
    CUDNN_DATA_DOUBLE = 1
} cudnnDataType_t;
cudnnStatus_t  cudnnCreateTensor4dDescriptor( cudnnTensor4dDescriptor_t *tensorDesc );
cudnnStatus_t  cudnnSetTensor4dDescriptorEx( cudnnTensor4dDescriptor_t tensorDesc,                          
                                                        cudnnDataType_t dataType, // image data type
                                                        int n,        // number of inputs (batch size)
                                                        int c,        // number of input feature maps
                                                        int h,        // height of input section
                                                        int w,        // width of input section
                                                        int nStride,
                                                        int cStride,
                                                        int hStride,
                                                        int wStride
                                                      );
cudnnStatus_t  cudnnDestroyTensor4dDescriptor( cudnnTensor4dDescriptor_t tensorDesc );
typedef enum
{
   CUDNN_ADD_IMAGE   = 0,
   CUDNN_ADD_SAME_HW = 0,        
   CUDNN_ADD_FEATURE_MAP = 1,
   CUDNN_ADD_SAME_CHW    = 1,
   CUDNN_ADD_SAME_C      = 2,
   CUDNN_ADD_FULL_TENSOR = 3
} cudnnAddMode_t;           
cudnnStatus_t  cudnnAddTensor4d( cudnnHandle_t handle,            
                                            cudnnAddMode_t mode,
                                            const void *alpha,                                         
                                            cudnnTensor4dDescriptor_t  biasDesc,
                                            const void                *biasData,
                                            cudnnTensor4dDescriptor_t  srcDestDesc,
                                            void                      *srcDestData
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
cudnnStatus_t  cudnnCreateFilterDescriptor( cudnnFilterDescriptor_t *filterDesc );
cudnnStatus_t  cudnnSetFilterDescriptor(    cudnnFilterDescriptor_t filterDesc,                                                       
                                                       cudnnDataType_t dataType, // image data type
                                                       int k,        // number of output feature maps
                                                       int c,        // number of input feature maps
                                                       int h,        // height of each input filter
                                                       int w         // width of  each input fitler
                                                  );
cudnnStatus_t  cudnnDestroyFilterDescriptor( cudnnFilterDescriptor_t filterDesc );
cudnnStatus_t  cudnnCreateConvolutionDescriptor( cudnnConvolutionDescriptor_t *convDesc );
cudnnStatus_t  cudnnSetConvolutionDescriptor(    cudnnConvolutionDescriptor_t convDesc,
                                                            cudnnTensor4dDescriptor_t inputTensorDesc,
                                                            cudnnFilterDescriptor_t filterDesc,
                                                            int pad_h,    // zero-padding height
                                                            int pad_w,    // zero-padding width
                                                            int u,        // vertical filter stride
                                                            int v,        // horizontal filter stride
                                                            int upscalex, // upscale the input in x-direction
                                                            int upscaley, // upscale the input in y-direction
                                                            cudnnConvolutionMode_t mode
                                                       );
cudnnStatus_t  cudnnGetOutputTensor4dDim( const cudnnConvolutionDescriptor_t convDesc,
                                                     cudnnConvolutionPath_t path,
                                                     int *n,
                                                     int *c,
                                                     int *h,
                                                     int *w
                                                   );
cudnnStatus_t  cudnnDestroyConvolutionDescriptor( cudnnConvolutionDescriptor_t convDesc );
typedef enum
{
    CUDNN_RESULT_ACCUMULATE      = 0,           /* Evaluate O += I * F */
    CUDNN_RESULT_NO_ACCUMULATE   = 1            /* Evaluate O = I * F  */
} cudnnAccumulateResult_t;
cudnnStatus_t  cudnnConvolutionForward(        cudnnHandle_t handle,
                                                          cudnnTensor4dDescriptor_t     srcDesc,
                                                          const void                   *srcData,
                                                          cudnnFilterDescriptor_t       filterDesc,
                                                          const void                   *filterData,
                                                          cudnnConvolutionDescriptor_t  convDesc,
                                                          cudnnTensor4dDescriptor_t     destDesc,
                                                          void                         *destData,
                                                          cudnnAccumulateResult_t       accumulate
                                                 );
cudnnStatus_t  cudnnConvolutionBackwardBias(   cudnnHandle_t handle,
                                                          cudnnTensor4dDescriptor_t     srcDesc,
                                                          const void                   *srcData,
                                                          cudnnTensor4dDescriptor_t     destDesc,
                                                          void                         *destData,
                                                          cudnnAccumulateResult_t       accumulate
                                                      );
cudnnStatus_t  cudnnConvolutionBackwardFilter( cudnnHandle_t handle,
                                                          cudnnTensor4dDescriptor_t     srcDesc,
                                                          const void                   *srcData,
                                                          cudnnTensor4dDescriptor_t     diffDesc,
                                                          const void                   *diffData, 
                                                          cudnnConvolutionDescriptor_t  convDesc,
                                                          cudnnFilterDescriptor_t       gradDesc,
                                                          void                         *gradData,
                                                          cudnnAccumulateResult_t       accumulate
                                                        );
cudnnStatus_t  cudnnConvolutionBackwardData( cudnnHandle_t handle,
                                                         cudnnFilterDescriptor_t       filterDesc,
                                                         const void                    *filterData,
                                                         cudnnTensor4dDescriptor_t     diffDesc,
                                                         const void                    *diffData,
                                                         cudnnConvolutionDescriptor_t  convDesc,
                                                         cudnnTensor4dDescriptor_t     gradDesc,
                                                         void                          *gradData,
                                                         cudnnAccumulateResult_t       accumulate
                                                       );                                                        
typedef enum
{
    CUDNN_POOLING_MAX     = 0,
    CUDNN_POOLING_AVERAGE = 1
} cudnnPoolingMode_t;
cudnnStatus_t  cudnnCreatePoolingDescriptor( cudnnPoolingDescriptor_t *poolingDesc);
cudnnStatus_t  cudnnSetPoolingDescriptor(    cudnnPoolingDescriptor_t poolingDesc,
                                                        cudnnPoolingMode_t mode,
                                                        int windowHeight,
                                                        int windowWidth,
                                                        int verticalStride,
                                                        int horizontalStride
                                                   );
cudnnStatus_t  cudnnGetPoolingDescriptor(    const cudnnPoolingDescriptor_t poolingDesc,
                                                        cudnnPoolingMode_t *mode,
                                                        int *windowHeight,
                                                        int *windowWidth,
                                                        int *verticalStride,
                                                        int *horizontalStride
                                                   );                                   
cudnnStatus_t  cudnnDestroyPoolingDescriptor( cudnnPoolingDescriptor_t poolingDesc );
cudnnStatus_t  cudnnPoolingForward(  cudnnHandle_t handle,
                                                cudnnPoolingDescriptor_t   poolingDesc,
                                                cudnnTensor4dDescriptor_t  srcDesc,
                                                const void                *srcData,
                                                cudnnTensor4dDescriptor_t  destDesc,
                                                void                      *destData
                                             );
cudnnStatus_t  cudnnPoolingBackward( cudnnHandle_t handle,
                                                cudnnPoolingDescriptor_t     poolingDesc,
                                                cudnnTensor4dDescriptor_t    srcDesc,
                                                const void                  *srcData,
                                                cudnnTensor4dDescriptor_t    srcDiffDesc,
                                                const void                  *srcDiffData,
                                                cudnnTensor4dDescriptor_t    destDesc,
                                                const void                  *destData,
                                                cudnnTensor4dDescriptor_t    destDiffDesc,
                                                void                        *destDiffData
                                              );
typedef enum
{
    CUDNN_ACTIVATION_SIGMOID = 0,
    CUDNN_ACTIVATION_RELU    = 1,
    CUDNN_ACTIVATION_TANH    = 2
} cudnnActivationMode_t;
cudnnStatus_t  cudnnActivationForward( cudnnHandle_t handle,
                                                  cudnnActivationMode_t mode,
                                                  cudnnTensor4dDescriptor_t  srcDesc,
                                                  const void                *srcData,
                                                  cudnnTensor4dDescriptor_t  destDesc,
                                                  void                      *destData
                                                );
cudnnStatus_t  cudnnActivationBackward( cudnnHandle_t handle,
                                                   cudnnActivationMode_t mode,
                                                   cudnnTensor4dDescriptor_t  srcDesc,
                                                   const void                *srcData,
                                                   cudnnTensor4dDescriptor_t  srcDiffDesc,
                                                   const void                *srcDiffData,
                                                   cudnnTensor4dDescriptor_t  destDesc,
                                                   const void                *destData,
                                                   cudnnTensor4dDescriptor_t  destDiffDesc,
                                                   void                      *destDiffData
                                                 );

]]

cudnn.C = ffi.load('libcudnn')
