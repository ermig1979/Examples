#pragma once

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cfloat>
#include <cmath>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>

__forceinline double time()
{
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return double(counter.QuadPart) / double(frequency.QuadPart);
}
#else
#include <sys/time.h>

inline __attribute__((always_inline)) double time()
{
    timeval t1;
    gettimeofday(&t1, NULL);
    return t1.tv_sec + t1.tv_usec * 0.000001;
}
#endif

struct mat_t
{
    float* p;
    int m, n;

    mat_t(int _m, int _n) : m(_m), n(_n), p((float*)_mm_malloc(_m * _n * 4, 64)) {}
    ~mat_t() { _mm_free(p); }
    int size() const { return m * n; }
};

typedef void (*gemm_t)(const mat_t& a, const mat_t& b, mat_t& c);

inline void init(mat_t& mat, float min, float max, size_t order = 1)
{
    float range = (max - min) / order;
    for (int i = 0, n = mat.size(); i < n; ++i)
    {
        float val = 0;
        for (int o = 0; o < order; ++o)
            val += float(rand()) / float(RAND_MAX);
        mat.p[i] = val * range + min;
    }
}

struct stat_t
{
    int count;
    double sum, sqsum, min, max;

    stat_t() 
    {
        count = 0;
        sum = 0;
        sqsum = 0;
        min = DBL_MAX;
        max = -DBL_MAX;
    }

    void update(double val)
    {
        count++;
        sum += val;
        sqsum += val * val;
        min = std::min(min, val);
        max = std::max(max, val);
    }

    std::string info(int precision)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(precision);
        ss << "{";
        ss << " min: " << min;
        ss << " max: " << max;
        ss << " avg: " << sum /count;
        ss << " std: " << sqrt(sqsum/count - sum * sum / count /count) ;
        ss << " }";
        return ss.str();
    }
};  

struct diff_t
{
    stat_t a, b, d;

    diff_t() { }
};

inline bool diff(const mat_t& a, const mat_t& b, diff_t & d)
{
    if (a.m != b.m || a.n != b.n)
        return false;

    for (int i = 0, n = a.size(); i < n; ++i)
    {
        d.a.update(a.p[i]);
        d.b.update(b.p[i]);
        d.d.update(a.p[i] - b.p[i]);
    }

    return true;
}