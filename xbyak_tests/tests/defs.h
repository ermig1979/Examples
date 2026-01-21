
#pragma once

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include "../xbyak/xbyak.h"
#include "../xbyak/xbyak_util.h"
#include "../xbyak/xbyak_bin2hex.h"

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>

__forceinline double Time()
{
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return double(counter.QuadPart) / double(frequency.QuadPart);
}
#else
#include <sys/time.h>

inline __attribute__((always_inline)) double Time()
{
    timeval t1;
    gettimeofday(&t1, NULL);
    return t1.tv_sec + t1.tv_usec * 0.000001;
}
#endif

//--------------------------------------------------------------------------------------------------

struct Buf
{
    float* p;
    int n;

    Buf(int size) : n(size), p((float*)_mm_malloc(size * 4, 64)) {}
    ~Buf() { _mm_free(p); }
};

typedef void (*GemmPtr)(int M, int N, int K, const float* A, const float* B, float* C);

inline void Init(Buf& buf)
{
    for (int i = 0; i < buf.n; ++i)
        buf.p[i] = float(rand()) / float(RAND_MAX) - 0.5f;
}

inline float Square(float x)
{
    return x * x;
}

inline float Invalid(float a, float b, float e2)
{
    float d2 = Square(a - b);
    return d2 > e2 && d2 > e2 * (a * a + b * b);
}

inline bool Check(const Buf& control, const Buf& current, const std::string& desc, float eps = 0.001f)
{
    assert(control.n == current.n);
    float e2 = Square(eps);
    for (int i = 0; i < control.n; ++i)
    {
        if (Invalid(control.p[i], current.p[i], e2))
        {
            std::cout << desc << " : check error at " << i << ": ";
            std::cout << std::setprecision(4) << std::fixed << control.p[i] << " != " << current.p[i] << std::endl;
            return false;
        }
    }
    return true;
}

//--------------------------------------------------------------------------------------------------

bool TestAdd2Ints();

bool TestAdd2Fp32Vecs();

bool TestStruct();

bool TestSgemm();

