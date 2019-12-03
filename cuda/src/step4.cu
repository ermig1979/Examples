#include "defs.h"
#include <vector_types.h>
#include <device_functions.h>

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

__global__ void gemm_v4d(int M, int N, int K, const float4* A, const float4* B, float4* C)
{
    int Ma = M / TS * TS;
    int Na = N / TS * TS;
    int Ka = K / TS;
    int i = TS * blockIdx.y + threadIdx.y;
    int j = PTS * blockIdx.x + threadIdx.x;
    if (i < M && j < N)
    {
        int k0 = 0;
        float4 _c = { 0, 0, 0, 0 };
        if (i < Ma && j < Na)
        {
            __shared__ float4 sA[TS][PTS];
            __shared__ float4 sB[TS][PTS];
            for (; k0 < Ka; k0 += 1)
            {
#if __CUDA_ARCH__ >= 320
                sA[threadIdx.y][threadIdx.x] = __ldg(&A[(i)*K / W + (k0 * PTS + threadIdx.x)]);
                sB[threadIdx.y][threadIdx.x] = __ldg(&B[(k0 * TS + threadIdx.y) * N / W + (j)]);
#else
                sA[threadIdx.y][threadIdx.x] = A[(i)*K / W + (k0 * PTS + threadIdx.x)];
                sB[threadIdx.y][threadIdx.x] = B[(k0 * TS + threadIdx.y) * N / W + (j)];
#endif
                __syncthreads();
                float4 _a, _b;
                float a;
                for (int k = 0; k < PTS; k++)
                {
                    _a = sA[threadIdx.y][k];
#pragma unroll
                    for (int wk = 0; wk < W; wk++)
                    {
                        switch (wk)
                        {
                        case 0: a = _a.x; break;
                        case 1: a = _a.y; break;
                        case 2: a = _a.z; break;
                        case 3: a = _a.w; break;
                        }
                        _b = sB[k * W + wk][threadIdx.x];
                        _c.x += a * _b.x;
                        _c.y += a * _b.y;
                        _c.z += a * _b.z;
                        _c.w += a * _b.w;
                    }
                }
                __syncthreads();
            }
        }
        C[(i)*N / W + j] = _c;
    }
}


int gemm_gpu_v4d(int M, int N, int K, const float* A, const float* B, float* C)
{
    dim3 grid(PTS, TS);
    dim3 block((N + TS - 1) / TS, (M + TS - 1) / TS);
    const int n = repeats(M, N, K, 0.21);
    for (int i = 0; i < n; ++i)
    {
        gemm_v4d <<<block, grid >>> (M, N, K, (const float4*)A, (const float4*)B, (float4*)C);
    }
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}