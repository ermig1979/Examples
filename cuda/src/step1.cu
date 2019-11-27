#include "defs.h"

__global__ void gemm_v1(int M, int N, int K, const float * A, const float * B, float * C)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < M && j < N)
    {
        float c = 0;
        for (int k = 0; k < K; ++k)
            c += A[i * K + k] * B[k * N + j];
        C[i * N + j] = c;
    }
}

int gemm_gpu_v1(int M, int N, int K, const float * A, const float * B, float * C)
{
    const int TS = 16;
    dim3 grid(TS, TS);
    dim3 block((N + TS - 1)/ TS, (M + TS - 1)/ TS);
    const int n = repeats(M, N, K, 0.050);
    for (int i = 0; i < n; ++i)
        gemm_v1<<<block, grid>>>(M, N, K, A, B, C);
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}