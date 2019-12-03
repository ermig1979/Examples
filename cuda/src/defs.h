#pragma once

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <cuda_runtime.h>

inline int repeats(int M, int N, int K, double k)
{
    int n = std::max(int(double(1024) / double(M) * double(1024) / double(N) * double(1024) / double(K) * k * 1000.0), 1);
    //std::cout << "repeats = " << n << std::endl;;
    return n;
}

typedef int (*gemm_t)(int M, int N, int K, const float * A, const float * B, float * C);

int gemm_cublas(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v0a(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v1a(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v2a(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v3a(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v4a(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v4b(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v4c(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v4d(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v5a(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v5b(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v5c(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v6a(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v7a(int M, int N, int K, const float* A, const float* B, float* C);

const int TRX = 16;
const int TRY = 16;

__global__ void transpose(int P, int Q, const float* src, float* dst);

struct gpu_buf_t
{
    float* p;
    int n;

    gpu_buf_t(int size)
        : n(size)
        , p(0)
    {
        cudaError_t error = cudaMalloc(&p, n * sizeof(float));
        assert(error == cudaSuccess);
    }

    ~gpu_buf_t()
    {
        if (p)
        {
            cudaError_t error = cudaFree(p);
            assert(error == cudaSuccess);
            p = 0;
        }
    }
};
