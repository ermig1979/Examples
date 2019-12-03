#include "defs.h"

const int TS = 32;
const int WPT = 8;
const int PTS = TS / WPT;

const int TSDK = 16;
const int LPT = TSDK * WPT / TS;

__global__ void transpose(int P, int Q, const float * src, float * dst) 
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    const int ID0 = blockIdx.x * TRX + tx;
    const int ID1 = blockIdx.y * TRY + ty;
    __shared__ float buf[TRX][TRY];
    if (ID0 < P && ID1 < Q)
        buf[ty][tx] = src[ID1 * P + ID0];
    __syncthreads();
    const int newID0 = blockIdx.y * TRY + tx;
    const int newID1 = blockIdx.x * TRX + ty;
    if (newID0 < Q && newID1 < P)
        dst[newID1 * Q + newID0] = buf[tx][ty];
}

__global__ void gemm_v5a(int M, int N, int K, const float* A, const float* B, float* C)
{
    int i0 = TS * blockIdx.y + threadIdx.y;
    int j = TS * blockIdx.x + threadIdx.x;
    float c[WPT];
    for (int w = 0; w < WPT; w++)
        c[w] = 0.0f;
    __shared__ float sA[TS][TS];
    __shared__ float sB[TS][TS];
    for (int k0 = 0; k0 < K; k0 += TS)
    {
        for (int w = 0; w < WPT; w++) 
        {
            sA[threadIdx.y + w * PTS][threadIdx.x] = A[(i0 + w * PTS) * K + (k0 + threadIdx.x)];
            sB[threadIdx.y + w * PTS][threadIdx.x] = B[(j) * K + (k0 + threadIdx.y + w * PTS)]; 
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
    for (int w = 0; w < WPT; w++)
        C[(i0 + w * PTS) * N + j] = c[w];
}

int gemm_gpu_v5a(int M, int N, int K, const float* A, const float* B, float* C)
{
    gpu_buf_t tB(N * K);
    dim3 gridT(TRX, TRY);
    dim3 blockT((N + TRX - 1) / TRX, (K + TRY - 1) / TRY);

    dim3 grid(TS, TS / WPT);
    dim3 block((N + TS - 1) / TS, (M + TS - 1) / TS);
    const int n = repeats(M, N, K, 0.370);
    for (int i = 0; i < n; ++i)
    {
        transpose <<<blockT, gridT >>> (N, K, B, tB.p);
        gemm_v5a <<<block, grid >>> (M, N, K, A, tB.p, C);
    }
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}

__global__ void gemm_v5b(int M, int N, int K, const float* A, const float* B, float* C)
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
                    sA[threadIdx.y + w * PTS][threadIdx.x] = A[(k0 + threadIdx.x) * M + (i0 + w * PTS)];
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

int gemm_gpu_v5b(int M, int N, int K, const float* A, const float* B, float* C)
{
    gpu_buf_t tA(M * K);
    dim3 gridT(TRX, TRY);
    dim3 blockT((M + TRX - 1) / TRX, (K + TRY - 1) / TRY);

    dim3 grid(TS, TS / WPT);
    dim3 block((N + TS - 1) / TS, (M + TS - 1) / TS);
    const int n = repeats(M, N, K, 0.370);
    for (int i = 0; i < n; ++i)
    {
        transpose <<<blockT, gridT >>> (M, K, A, tA.p);
        gemm_v5b <<<block, grid >>> (M, N, K, tA.p, B, C);
    }
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}

__global__ void gemm_v5c(int M, int N, int K, const float* A, const float* B, float* C)
{
    int i0 = TS * blockIdx.y + threadIdx.y;
    int j = TS * blockIdx.x + threadIdx.x;
    float c[WPT];
    for (int w = 0; w < WPT; w++)
        c[w] = 0.0f;
    __shared__ float sA[TSDK][TS];
    __shared__ float sB[TSDK][TS];
    for (int k0 = 0; k0 < K; k0 += TSDK)
    {
        for (int l = 0; l < LPT; l++)
        {
            int k = k0 + threadIdx.y + l * PTS;
            sA[threadIdx.y + l * PTS][threadIdx.x] = A[(k) * M + (TS * blockIdx.y + threadIdx.x)];
            sB[threadIdx.y + l * PTS][threadIdx.x] = B[(k) * N + (TS * blockIdx.x + threadIdx.x)];
        }
        __syncthreads();
        for (int k = 0; k < TSDK; k++)
        {
            float b = sB[k][threadIdx.x];
            for (int w = 0; w < WPT; w++)
                c[w] += b * sA[k][threadIdx.y + w * PTS];
        }
        __syncthreads();
    }
    for (int w = 0; w < WPT; w++)
        C[(i0 + w * PTS) * N + j] = c[w];
}

int gemm_gpu_v5c(int M, int N, int K, const float* A, const float* B, float* C)
{
    gpu_buf_t tA(M * K);
    dim3 gridT(TRX, TRY);
    dim3 blockT((M + TRX - 1) / TRX, (K + TRY - 1) / TRY);

    dim3 grid(TS, TS / WPT);
    dim3 block((N + TS - 1) / TS, (M + TS - 1) / TS);
    const int n = repeats(M, N, K, 0.370);
    for (int i = 0; i < n; ++i)
    {
        transpose<<<blockT, gridT>>>(M, K, A, tA.p);
        gemm_v5c<<<block, grid>>>(M, N, K, tA.p, B, C);
    }
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}
