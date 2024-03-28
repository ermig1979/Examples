#pragma once

#include "defs.h"

union F32 
{
    F32(float val) : f32{ val }  {   }
    F32(uint32_t val) : u32{ val } {  }

    float f32;
    uint32_t u32;
};

inline float bf16_original(float val)
{
    return val;
}

inline float bf16_nearest(float val)
{
    return F32((F32(val).u32 + 0x8000) & 0xFFFF0000).f32;
}

struct bf16_original_t
{
    static inline float convert(float val)
    {
        return val;
    }
};

struct bf16_nearest_t
{
    static inline float convert(float val)
    {
        return F32((F32(val).u32 + 0x8000) & 0xFFFF0000).f32;
    }
};

void gemm_control(const mat_t& a, const mat_t& b, mat_t& c)
{
    assert(a.m == c.m && a.n == b.m && b.n == c.n);
    for (int i = 0; i < a.m; ++i)
    {
        for (int j = 0; j < b.n; ++j)
        {
            double sum = 0;
            for (int k = 0; k < a.n; ++k)
            {
                double _a = a.p[i * a.n + k];
                double _b = b.p[k * b.n + j];
                sum += _a * _b;
            }
            c.p[i * b.n + j] = (float)sum;
        }
    }
}

template<typename C> void gemm(const mat_t& a, const mat_t& b, mat_t& c)
{
    assert(a.m == c.m && a.n == b.m && b.n == c.n);
    for (int i = 0; i < a.m; ++i)
    {
        for (int j = 0; j < b.n; ++j)
        {
            float sum = 0;
            for (int k = 0; k < a.n; ++k)
            {
                float _a = C::convert(a.p[i * a.n + k]);
                float _b = C::convert(b.p[k * b.n + j]);
                sum += _a * _b;
            }
            c.p[i * b.n + j] = sum;
        }
    }
}