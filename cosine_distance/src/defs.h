#pragma once

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <vector>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
namespace cs
{
    __forceinline double Time()
    {
        LARGE_INTEGER counter, frequency;
        QueryPerformanceCounter(&counter);
        QueryPerformanceFrequency(&frequency);
        return double(counter.QuadPart) / double(frequency.QuadPart);
    }
}
#else
#include <sys/time.h>
namespace cs
{
    inline __attribute__((always_inline)) double Time()
    {
        timeval t1;
        gettimeofday(&t1, NULL);
        return t1.tv_sec + t1.tv_usec * 0.000001;
    }
}
#endif

namespace cs
{
    typedef std::vector<float> Buffer32f;
    typedef std::vector<uint32_t> Buffer32u;
    typedef std::vector<uint8_t> Buffer8u;
    typedef std::string String;
}


