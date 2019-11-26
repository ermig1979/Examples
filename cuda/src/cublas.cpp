#include "defs.h"

#include <cublas_v2.h>

int gemm_cublas(int M, int N, int K, const float * A, const float * B, float * C)
{
    const float alpha = 1.0f, beta = 0.0f;
    const int n = repeats(M, N, K, 0.800);
    cublasHandle_t handle;
    assert(cublasCreate(&handle) == cudaSuccess);
    for(int i = 0; i < n; ++i)
        assert(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, B, N, A, K, &beta, C, N) == CUBLAS_STATUS_SUCCESS);
    assert(cublasDestroy(handle) == cudaSuccess);
    return n;
}