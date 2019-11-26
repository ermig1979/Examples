#include "defs.h"

__global__ void gemm_v0(int M, int N, int K, const float * A, const float * B, float * C)
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

int gemm_gpu_v0(int M, int N, int K, const float * A, const float * B, float * C)
{
    const int n = repeats(M, N, K, 0.030);
    const int S = 16;
    dim3 grid(S, S);
    dim3 block((N + S - 1)/S, (M + S - 1)/S);
    for (int i = 0; i < n; ++i)
        gemm_v0<<<block, grid>>>(M, N, K, A, B, C);
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}