#include "defs.h"

#define S 32

__global__ void gemm_v2(int M, int N, int K, const float * A, const float * B, float * C)
{
    int Ma = M / S * S;
    int Na = N / S * S;
    int Ka = K / S * S;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < M && j < N)
    {
        int kB = 0;
        float c = 0;
        if(i < Ma && j < Na)
        {
            for (; kB < Ka; kB += S) 
            {
                __shared__ float sA[S][S];
                __shared__ float sB[S][S];
                sA[threadIdx.y][threadIdx.x] = A[i * K + (kB + threadIdx.x)];
                sB[threadIdx.y][threadIdx.x] = B[(kB + threadIdx.y) * N + j];
                __syncthreads();
                for (int k = 0; k < S; ++k)
                    c += sA[threadIdx.y][k] * sB[k][threadIdx.x];
                __syncthreads();
            }
        }
        for (int k = kB; k < K; ++k)
            c += A[i * K + k] * B[k * N + j];
        C[i * N + j] = c;
    }
}

int gemm_gpu_v2(int M, int N, int K, const float * A, const float * B, float * C)
{
    dim3 grid(S, S);
    dim3 block((N + S - 1)/S, (M + S - 1)/S);
    const int n = repeats(M, N, K, 0.170);
    for (int i = 0; i < n; ++i)
        gemm_v2<<<block, grid>>>(M, N, K, A, B, C);
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}