#include "defs.h"

const int TS = 32;

__global__ void gemm_v2(int M, int N, int K, const float * A, const float * B, float * C)
{
    int Ma = M / TS * TS;
    int Na = N / TS * TS;
    int Ka = K / TS * TS;
    int i = TS * blockIdx.y + threadIdx.y;
    int j = TS * blockIdx.x + threadIdx.x;
    if(i < M && j < N)
    {
        int k0 = 0;
        float c = 0;
        if(i < Ma && j < Na)
        {
            __shared__ float sA[TS][TS];
            __shared__ float sB[TS][TS];
            for (; k0 < Ka; k0 += TS)
            {
                sA[threadIdx.y][threadIdx.x] = A[i * K + (k0 + threadIdx.x)];
                sB[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + j];
                __syncthreads();
                for (int k = 0; k < TS; ++k)
                    c += sA[threadIdx.y][k] * sB[k][threadIdx.x];
                __syncthreads();
            }
        }
        for (int k = k0; k < K; ++k)
            c += A[i * K + k] * B[k * N + j];
        C[i * N + j] = c;
    }
}

int gemm_gpu_v2(int M, int N, int K, const float * A, const float * B, float * C)
{
    dim3 grid(TS, TS);
    dim3 block((N + TS - 1)/ TS, (M + TS - 1)/ TS);
    const int n = repeats(M, N, K, 0.170);
    for (int i = 0; i < n; ++i)
        gemm_v2<<<block, grid>>>(M, N, K, A, B, C);
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}