
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

//--------------------------------------------------------------------------------------------------

bool TestAdd2Ints();

bool TestAdd2Fp32Vecs();

