#pragma once

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <cuda_runtime.h>

typedef void (*gemm_t)(int M, int N, int K, const float * A, const float * B, float * C);

void gemm_gpu_v0(int M, int N, int K, const float * A, const float * B, float * C);

struct cpu_buf_t
{
    float * p;
    int n;

    cpu_buf_t(int size)
        : n(size)
        , p((float*)_mm_malloc(size * 4, 64))
    {
    }

    ~cpu_buf_t()
    {
        _mm_free(p);
    }
};

struct gpu_buf_t
{
    float *p;
    int n;

    gpu_buf_t(int size)
        : n(size)
        , p(0)
    {
        cudaError_t error = cudaMalloc(&p, n*sizeof(float));
        assert(error == cudaSuccess);
    }

    ~gpu_buf_t()
    {
        if(p)
        {
            cudaError_t error = cudaFree(p);
            assert(error == cudaSuccess);
            p = 0;
        }
    }
};

void copy(const cpu_buf_t & src, gpu_buf_t & dst);
void copy(const gpu_buf_t & src, cpu_buf_t & dst);
