#include "defs.h"

__global__ void gemm_v0(int M, int N, int K, const float * A, const float * B, float * C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M)
    {
        float * c = C + i * N;
        for (int j = 0; j < N; ++j)
            c[j] = 0;
        for (int k = 0; k < K; ++k)
        {
            const float * b = B + k * N;
            float a = A[i*K + k];
            for (int j = 0; j < N; ++j)
                c[j] += a * b[j];
        }
    }
}

void gemm_gpu_v0(int M, int N, int K, const float * A, const float * B, float * C)
{
    int threadsPerBlock = 256;
    int blocksPerGrid =(M + threadsPerBlock - 1) / threadsPerBlock;
    //set_c_zero<<<blocksPerGrid, threadsPerBlock>>>(C, M, N);
    gemm_v0<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, A, B, C);
    assert(cudaGetLastError() == cudaSuccess);
}