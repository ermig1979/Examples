#include "defs.h"

__global__ void gemm_v0a(int M, int N, int K, const float * A, const float * B, float * C)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < M && j < N)
    {
        C[i * N + j] = 0;
        for (int k = 0; k < K; ++k)
            C[i * N + j] += A[i * K + k] * B[k * N + j];
    }
}

int gemm_gpu_v0a(int M, int N, int K, const float * A, const float * B, float * C)
{
    const int n = repeats(M, N, K, 0.03);
    const int TS = 16;
    dim3 grid(TS, TS);
    dim3 block((N + TS - 1)/TS, (M + TS - 1)/TS);
    for (int i = 0; i < n; ++i)
        gemm_v0a<<<block, grid>>>(M, N, K, A, B, C);
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}