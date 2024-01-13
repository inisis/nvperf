#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp8.h>

#define WARMUP_ITER 10
#define ITER 50

#define CHECK_CUDA_STATUS(call)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));                        \
            return -1;                                                                                             \
        }                                                                                                              \
    } while (0)


#define CHECK_CUBLAS_STATUS(call)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t status = call;                                                                                   \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            printf("cuBLAS API failed with status %d: %s\n", status, cublasGetStatusString(status));                   \
            return -1;                                                                                             \
        }                                                                                                              \
    } while (0)

struct GpuTimer
{
  cudaEvent_t start_;
  cudaEvent_t stop_;

  GpuTimer()
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void Start()
  {
    cudaEventRecord(start_, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop_, 0);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed, start_, stop_);
    return elapsed;
  }
};


double float32_perf(bool use_tensorcode)
{
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    int m = 16384;
    int n = 16384;
    int k = 16384;
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS_STATUS(cublasCreate(&handle));

    if(use_tensorcode==true)
    {
        CHECK_CUBLAS_STATUS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    }

    int lda = (transa == CUBLAS_OP_N) ? max (1, m) : max (1, k);
    int ldb = (transb == CUBLAS_OP_N) ? max (1, k) : max (1, n);
    int ldc = max (1, m);
    int ka = (transa == CUBLAS_OP_N) ? k : m;
    int kb = (transb == CUBLAS_OP_N) ? n : k;
    
    size_t Asz = (size_t)lda * ka * sizeof (float);
    size_t Bsz = (size_t)ldb * kb * sizeof (float);
    size_t Csz = (size_t)ldc * n  * sizeof (float);
    float *A_d = 0, *B_d = 0, *C_d = 0;
    CHECK_CUDA_STATUS(cudaMalloc((void**)&A_d, Asz));
    CHECK_CUDA_STATUS(cudaMalloc((void**)&B_d, Bsz));
    CHECK_CUDA_STATUS(cudaMalloc((void**)&C_d, Csz));

    GpuTimer timer;

    // Warm-up phase
    for (int warmup_iter = 0; warmup_iter < WARMUP_ITER; ++warmup_iter) {
        CHECK_CUBLAS_STATUS(cublasSgemm(handle, transa, transb, m, n, k, &alpha, A_d, lda, B_d, ldb, &beta, C_d, ldc));
        cudaDeviceSynchronize();
    }

    // Actual measurement phase
    timer.Start();
    for (int iter = 0; iter < ITER; ++iter) {
        CHECK_CUBLAS_STATUS(cublasSgemm(handle, transa, transb, m, n, k, &alpha, A_d, lda, B_d, ldb, &beta, C_d, ldc));
        cudaDeviceSynchronize();
    }
    timer.Stop();

    auto elapsed = timer.Elapsed() / ITER;
    auto tflop = 2.0e-9 * m * n * k / elapsed;

    return tflop;
}


double float16_perf()
{
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    int m = 16384;
    int n = 16384;
    int k = 16384;
    __half alpha = 1.0f;
    __half beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS_STATUS(cublasCreate(&handle));
    CHECK_CUBLAS_STATUS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    int lda = (transa == CUBLAS_OP_N) ? max (1, m) : max (1, k);
    int ldb = (transb == CUBLAS_OP_N) ? max (1, k) : max (1, n);
    int ldc = max (1, m);
    int ka = (transa == CUBLAS_OP_N) ? k : m;
    int kb = (transb == CUBLAS_OP_N) ? n : k;
    
    size_t Asz = (size_t)lda * ka * sizeof (__half);
    size_t Bsz = (size_t)ldb * kb * sizeof (__half);
    size_t Csz = (size_t)ldc * n  * sizeof (__half);
    __half *A_d = 0, *B_d = 0, *C_d = 0;
    CHECK_CUDA_STATUS(cudaMalloc((void**)&A_d, Asz));
    CHECK_CUDA_STATUS(cudaMalloc((void**)&B_d, Bsz));
    CHECK_CUDA_STATUS(cudaMalloc((void**)&C_d, Csz));

    for (int warmup_iter = 0; warmup_iter < WARMUP_ITER; ++warmup_iter) {
        CHECK_CUBLAS_STATUS(cublasHgemm(handle, transa, transb, m, n, k, &alpha, A_d, lda, B_d, ldb, &beta, C_d, ldc));
        cudaDeviceSynchronize();
    }

    GpuTimer timer;
    // Actual measurement phase
    timer.Start();
    for (int iter = 0; iter < ITER; ++iter) {
        CHECK_CUBLAS_STATUS(cublasHgemm(handle, transa, transb, m, n, k, &alpha, A_d, lda, B_d, ldb, &beta, C_d, ldc));
        cudaDeviceSynchronize();
    }
    timer.Stop();

    auto elapsed = timer.Elapsed() / ITER;
    auto tflop = 2.0e-9 * m * n * k / elapsed;

    return tflop;
}


double int8_perf()
{
    using mt = char;
    using rt = int;
    using st = int;
    cudaDataType   Atype = CUDA_R_8I;
    cudaDataType   Ctype = CUDA_R_32I;
    cublasComputeType_t   computeType = CUBLAS_COMPUTE_32I;

    int dim = 16384;
    int m = dim;
    int n = dim;
    int k = dim;
    mt *A, *B;
    rt *C;
    CHECK_CUDA_STATUS(cudaMalloc(&A, sizeof(A[0])*m*k));
    CHECK_CUDA_STATUS(cudaMalloc(&B, sizeof(B[0])*n*k));
    CHECK_CUDA_STATUS(cudaMalloc(&C, sizeof(C[0])*m*n));
    st alpha = 1;
    st beta = 0;
    cublasHandle_t h;
    cublasStatus_t stat = cublasCreate(&h);

    for (int warmup_iter = 0; warmup_iter < WARMUP_ITER; ++warmup_iter) {
        CHECK_CUBLAS_STATUS(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, Atype, dim, 
                        B, Atype, dim, &beta, C, Ctype, dim, computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cudaDeviceSynchronize();
    }

    GpuTimer timer;
    // Actual measurement phase
    timer.Start();
    for (int iter = 0; iter < ITER; ++iter) {
        CHECK_CUBLAS_STATUS(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, Atype, dim, 
                        B, Atype, dim, &beta, C, Ctype, dim, computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cudaDeviceSynchronize();
    }
    timer.Stop();

    auto elapsed = timer.Elapsed() / ITER;
    auto tflop = 2.0e-9 * m * n * k / elapsed;

    return tflop;
}


double fp8_perf()
{   
    float alpha = 2.0, beta = 0.0;
    float *a_scale, *b_scale, *c_scale, *d_scale, *amax_d;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    __nv_fp8_e4m3 *A, *B, *D;
    int dim = 16384;
    int m = dim;
    int n = dim;
    int k = dim;    
    int lda = (transa == CUBLAS_OP_N) ? max (1, m) : max (1, k);
    int ldb = (transb == CUBLAS_OP_N) ? max (1, k) : max (1, n);
    int ldc = max (1, m);    
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    void *workspace;
    size_t workspaceSize = 12ULL * 1024 * 1024 * 1024;
    cublasLtHandle_t ltHandle;
    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    CHECK_CUBLAS_STATUS(cublasLtCreate(&ltHandle));
    CHECK_CUDA_STATUS(cudaMalloc(reinterpret_cast<void**>(&A), m * k * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA_STATUS(cudaMalloc(reinterpret_cast<void**>(&B), n * k * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA_STATUS(cudaMalloc(reinterpret_cast<void**>(&D), m * n * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA_STATUS(cudaMalloc(&workspace, workspaceSize));
    CHECK_CUDA_STATUS(cudaMalloc(reinterpret_cast<void**>(&a_scale), sizeof(*a_scale)));
    CHECK_CUDA_STATUS(cudaMalloc(reinterpret_cast<void**>(&b_scale), sizeof(*b_scale)));
    CHECK_CUDA_STATUS(cudaMalloc(reinterpret_cast<void**>(&c_scale), sizeof(*c_scale)));
    CHECK_CUDA_STATUS(cudaMalloc(reinterpret_cast<void**>(&d_scale), sizeof(*d_scale)));
    CHECK_CUDA_STATUS(cudaMalloc(reinterpret_cast<void**>(&amax_d), sizeof(*amax_d)));

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    CHECK_CUBLAS_STATUS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    // set scaling factors
    CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
    CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale, sizeof(c_scale)));
    CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
    CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d, sizeof(amax_d)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    // table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));
    CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_8F_E4M3, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    CHECK_CUBLAS_STATUS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS_STATUS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    CHECK_CUBLAS_STATUS(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        CHECK_CUBLAS_STATUS(CUBLAS_STATUS_NOT_SUPPORTED);
    }


    for (int warmup_iter = 0; warmup_iter < WARMUP_ITER; ++warmup_iter) {
        CHECK_CUBLAS_STATUS(cublasLtMatmul(ltHandle, operationDesc, &alpha, A, Adesc, B, Bdesc, &beta,
                          nullptr, Cdesc, D, Ddesc, &heuristicResult.algo,
                          workspace, workspaceSize, 0));
        cudaDeviceSynchronize();
    }

    GpuTimer timer;
    // Actual measurement phase
    timer.Start();
    for (int iter = 0; iter < ITER; ++iter) {
        CHECK_CUBLAS_STATUS(cublasLtMatmul(ltHandle, operationDesc, &alpha, A, Adesc, B, Bdesc, &beta,
                          nullptr, Cdesc, D, Ddesc, &heuristicResult.algo,
                          workspace, workspaceSize, 0));
        cudaDeviceSynchronize();
    }
    timer.Stop();

    auto elapsed = timer.Elapsed() / ITER;
    auto tflop = 2.0e-9 * m * n * k / elapsed;

    return tflop;
}


extern "C" bool runPerf(int device_id){ 
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    printf("\nDevice %d: \"%s\"\n", device_id, deviceProp.name);

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);    
    
    printf(
    "  GPU Max Clock rate:                            %.0f MHz\n",
    deviceProp.clockRate * 1e-3f);

#if CUDART_VERSION >= 5000
    // This is supported in CUDA 5.0 (runtime API device properties)
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
#else
    // This only available in CUDA 4.0-4.2 (but these were only exposed in the
    // CUDA Driver API)
    int memoryClock;
    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                          dev);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           memoryClock * 1e-3f);
#endif

    printf("fp32 without tensorcore: %f TFLOPS\n", float32_perf(false));
    printf("fp32 with    tensorcore: %f TFLOPS\n", float32_perf(true));
    printf("fp16 with    tensorcore: %f TFLOPS\n", float16_perf());
    printf("int8 with    tensorcore: %f TFLOPS\n", int8_perf());
    printf("fp8  with    tensorcore: %f TFLOPS\n", fp8_perf());

    return 0;
}
