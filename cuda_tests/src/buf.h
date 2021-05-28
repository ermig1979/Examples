#pragma once

#include "defs.h"

struct gpu_buf_t
{
    float* p;
    int n;

    gpu_buf_t(int size)
        : n(size)
        , p(0)
    {
        CHECK(cudaMalloc(&p, n * sizeof(float)));
    }

    ~gpu_buf_t()
    {
        if (p)
        {
            CHECK(cudaFree(p));
            p = 0;
        }
    }
};

struct cpu_buf_t
{
    float* p;
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

inline void copy(const cpu_buf_t& src, gpu_buf_t& dst)
{
    assert(src.n == dst.n);
    CHECK(cudaMemcpy(dst.p, src.p, src.n * sizeof(float), cudaMemcpyHostToDevice));
}

inline void copy(const gpu_buf_t& src, cpu_buf_t& dst)
{
    assert(src.n == dst.n);
    CHECK(cudaMemcpy(dst.p, src.p, src.n * sizeof(float), cudaMemcpyDeviceToHost));
}

