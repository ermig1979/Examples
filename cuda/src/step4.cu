#include "defs.h"

const int TS = 32;
const int W = 4;
const int PTS = TS / W;

typedef float vec_t[W];

__global__ void gemm_v4a(int M, int N, int K, const float* A, const float* B, float* C)
{
    int Ma = M / TS * TS;
    int Na = N / TS * TS;
    int Ka = K / TS * TS;
    int i = TS * blockIdx.y + threadIdx.y;
    int j0 = TS * blockIdx.x + threadIdx.x*W;
    if (i < M && j0 < N)
    {
        int k0 = 0;
        vec_t c;
        for (int w = 0; w < W; w++)
            c[w] = 0.0f;
        if (i < Ma && j0 < Na)
        {
            __shared__ float sA[TS][TS];
            __shared__ float sB[TS][TS];
            for (; k0 < Ka; k0 += TS)
            {
                for (int w = 0; w < W; w++)
                {
                    sA[threadIdx.y][threadIdx.x * W + w] = A[(i) * K + (k0 + threadIdx.x*W + w)];
                    sB[threadIdx.y][threadIdx.x * W + w] = B[(k0 + threadIdx.y) * N + (j0 + w)];
                }
                __syncthreads();
                vec_t _a, _b;
                float a;
                for (int k = 0; k < PTS; k++)
                {
                    for (int wk = 0; wk < W; wk++)
                        _a[wk] = sA[threadIdx.y][k * W + wk];
                    for (int wk = 0; wk < W; wk++)
                    {
                        a = _a[wk];
                        for (int w = 0; w < W; w++)
                            _b[w] = sB[k * W + wk][threadIdx.x * W + w];
                        for (int w = 0; w < W; w++)
                            c[w] += a * _b[w];
                    }
                }
                __syncthreads();
            }
        }
        for (int w = 0; w < W; w++)
            C[(i) * N + (j0 + w)] = c[w];
    }
}

int gemm_gpu_v4a(int M, int N, int K, const float* A, const float* B, float* C)
{
    dim3 grid(PTS, TS);
    dim3 block((N + TS - 1) / TS, (M + TS - 1) / TS);
    const int n = repeats(M, N, K, 0.21);
    for (int i = 0; i < n; ++i)
    {
        gemm_v4a<<<block, grid>>>(M, N, K, A, B, C);
    }
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}

__global__ void gemm_v4b(int M, int N, int K, const float* A, const float* B, float* C)
{
    int Ma = M / TS * TS;
    int Na = N / TS * TS;
    int Ka = K / TS * TS;
    int i = TS * blockIdx.y + threadIdx.y;
    int j0 = TS * blockIdx.x + threadIdx.x;
    if (i < M && j0 < N)
    {
        int k0 = 0;
        vec_t c;
        for (int w = 0; w < W; w++)
            c[w] = 0.0f;
        if (i < Ma && j0 < Na)
        {
            __shared__ float sA[TS][TS];
            __shared__ float sB[TS][TS];
            for (; k0 < Ka; k0 += TS)
            {
                for (int w = 0; w < W; w++)
                {
                    sA[threadIdx.y][threadIdx.x + PTS * w] = A[(i)*K + (k0 + threadIdx.x + PTS * w)];
                    sB[threadIdx.y][threadIdx.x + PTS * w] = B[(k0 + threadIdx.y) * N + (j0 + PTS * w)];
                }
                __syncthreads();
                vec_t _a, _b;
                float a;
                for (int k = 0; k < PTS; k++)
                {
                    for (int wk = 0; wk < W; wk++)
                        _a[wk] = sA[threadIdx.y][k * W + wk];
                    for (int wk = 0; wk < W; wk++)
                    {
                        a = _a[wk];
                        for (int w = 0; w < W; w++)
                            _b[w] = sB[k * W + wk][threadIdx.x + PTS * w];
                        for (int w = 0; w < W; w++)
                            c[w] += a * _b[w];
                    }
                }
                __syncthreads();
            }
        }
        for (int w = 0; w < W; w++)
            C[(i)*N + (j0 + PTS * w)] = c[w];
    }
}

int gemm_gpu_v4b(int M, int N, int K, const float* A, const float* B, float* C)
{
    dim3 grid(PTS, TS);
    dim3 block((N + TS - 1) / TS, (M + TS - 1) / TS);
    const int n = repeats(M, N, K, 0.21);
    for (int i = 0; i < n; ++i)
    {
        gemm_v4b<<<block, grid>>> (M, N, K, A, B, C);
    }
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}


struct floats
{
    float val[W];
};

__global__ void gemm_v4c(int M, int N, int K, const floats * A, const floats * B, floats * C)
{
    int Ma = M / TS * TS;
    int Na = N / TS * TS;
    int Ka = K / TS;
    int i = TS * blockIdx.y + threadIdx.y;
    int j = PTS * blockIdx.x + threadIdx.x;
    if (i < M && j < N)
    {
        int k0 = 0;
        floats _c;
        for (int w = 0; w < W; w++)
            _c.val[w] = 0.0f;
        if (i < Ma && j < Na)
        {
            __shared__ floats sA[TS][PTS];
            __shared__ floats sB[TS][PTS];
            for (; k0 < Ka; k0 += 1)
            {
                sA[threadIdx.y][threadIdx.x] = A[(i)*K/W + (k0*PTS + threadIdx.x)];
                sB[threadIdx.y][threadIdx.x] = B[(k0*TS + threadIdx.y) * N/W + (j)];
                __syncthreads();
                floats _a, _b;
                float a;
                for (int k = 0; k < PTS; k++)
                {
                    _a = sA[threadIdx.y][k];
                    for (int wk = 0; wk < W; wk++)
                    {
                        a = _a.val[wk];
                        _b = sB[k * W + wk][threadIdx.x];
                        for (int w = 0; w < W; w++)
                            _c.val[w] += a * _b.val[w];
                            //_c.val[w] = __fmaf_ru(a, _b.val[w], _c.val[w]);
                    }
                }
                __syncthreads();
            }
        }
        C[(i)*N/W + j] = _c;
    }
}


int gemm_gpu_v4c(int M, int N, int K, const float * A, const float * B, float * C)
{
    dim3 grid(PTS, TS);
    dim3 block((N + TS - 1)/ TS, (M + TS - 1)/ TS);
    const int n = repeats(M, N, K, 0.21);
    for (int i = 0; i < n; ++i)
    {
        gemm_v4c<<<block, grid>>> (M, N, K, (const floats*)A, (const floats*)B, (floats*)C);
    }
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}