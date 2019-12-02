#include "defs.h"

const int TS = 32;
const int WPT = 8;
const int PTS = TS / WPT;

__global__ void gemm_v3a(int M, int N, int K, const float* A, const float* B, float* C)
{
    int Ma = M / TS * TS;
    int Na = N / TS * TS;
    int Ka = K / TS * TS;
    int i0 = TS * blockIdx.y + threadIdx.y;
    int j = TS * blockIdx.x + threadIdx.x;
    if (i0 < M && j < N)
    {
        int k0 = 0;
        float c[WPT];
        for (int w = 0; w < WPT; w++)
            c[w] = 0.0f;
        if (i0 < Ma && j < Na)
        {
            __shared__ float sA[TS][TS];
            __shared__ float sB[TS][TS];
            for (; k0 < Ka; k0 += TS)
            {
                for (int w = 0; w < WPT; w++) 
                {
                    sA[threadIdx.y + w * PTS][threadIdx.x] = A[(i0 + w * PTS) * K + (k0 + threadIdx.x)];
                    sB[threadIdx.y + w * PTS][threadIdx.x] = B[(k0 + threadIdx.y + w * PTS) * N + (j)];
                }
                __syncthreads();
                for (int k = 0; k < TS; ++k)
                {
                    float b = sB[k][threadIdx.x];
                    for (int w = 0; w < WPT; w++)
                        c[w] += sA[threadIdx.y + w * PTS][k] * b;
                }
                __syncthreads();
            }
        }
        for (int w = 0; w < WPT; w++)
            C[(i0 + w * PTS) * N + j] = c[w];
    }
}

int gemm_gpu_v3a(int M, int N, int K, const float * A, const float * B, float * C)
{
    dim3 grid(TS, TS / WPT);
    dim3 block((N + TS - 1)/ TS, (M + TS - 1)/ TS);
    const int n = repeats(M, N, K, 0.370);
    for (int i = 0; i < n; ++i)
        gemm_v3a<<<block, grid>>>(M, N, K, A, B, C);
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}