#pragma once

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

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

struct vec_t
{
    float* p;
    int n;

    vec_t(int _n) : n(_n), p((float*)_mm_malloc(_n * 4, 64)) {}
    ~vec_t() { _mm_free(p); }
};

typedef void (*func_t)(const vec_t& src, vec_t& dst);

inline void init(vec_t& vec, float min, float max, bool rnd = false)
{
    if (rnd)
    {
        float range = max - min;
        for (int i = 0; i < vec.n; ++i)
        {
            float val = float(rand()) / float(RAND_MAX);
            vec.p[i] = val * range + min;
        }
    }
    else
    {
        float delta = (max - min) / (float)std::max(1, vec.n - 1);
        for (int i = 0; i < vec.n; ++i)
        {
            vec.p[i] = i * delta + min;
        }
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

inline bool diff(const vec_t& a, const vec_t& b, diff_t & d)
{
    if (a.n != b.n)
        return false;

    for (int i = 0; i < a.n; ++i)
    {
        d.a.update(a.p[i]);
        d.b.update(b.p[i]);
        d.d.update(a.p[i] - b.p[i]);
    }

    return true;
}