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

void gemm_gpu_v1(int M, int N, int K, const float * A, const float * B, float * C)
{
    const int n = 16;
    dim3 grid(n, n);
    dim3 block((N + n - 1)/n, (M + n - 1)/n);
    gemm_v1<<<block, grid>>>(M, N, K, A, B, C);
    assert(cudaGetLastError() == cudaSuccess);
}