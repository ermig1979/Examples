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
#include <cstring>

//-------------------------------------------------------------------------------------------------

#if defined(_MSC_VER)
#define NOMINMAX
#include <windows.h>

__forceinline double Time()
{
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return double(counter.QuadPart) / double(frequency.QuadPart);
}
#elif defined(__linux__)
#include <sys/time.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>

inline __attribute__((always_inline)) double Time()
{
    timeval t1;
    gettimeofday(&t1, NULL);
    return t1.tv_sec + t1.tv_usec * 0.000001;
}
#endif

inline std::string ToStr(std::string value, int width)
{
    std::stringstream ss;
    ss << std::setfill(' ') << std::setw(width) << value;
    return ss.str();
}

inline int Min(int a, int b)
{
    return a < b ? a : b;
}

inline int AlignLo(int value, int align)
{
    return value / align * align;
}

inline size_t Cache(int level)
{
    switch (level)
    {
    case 1: return 48 * 1024;
    case 2: return 1 * 1024 * 1024;
    case 3: return 1 * 1024 * 1024;
    default:
        return 0;
    }
}

inline int CurrentCore()
{
#if defined(__linux__)
    return sched_getcpu();
#endif
    return -1;
}

inline uint64_t CurrentFrequency()
{
#if defined(__linux__)
    int core = sched_getcpu();
    std::stringstream args;
    args << "sudo cat /sys/devices/system/cpu/cpu" << core << "/cpufreq/scaling_cur_freq";
    ::FILE* p = ::popen(args.str().c_str(), "r");
    if (p)
    {
        char buffer[1024];
        while (::fgets(buffer, 1024, p));
        ::pclose(p);
        return ::atoi(buffer) * uint64_t(1000);
    }
#endif
    return 0;
}

inline void PrintCurrentFrequency()
{
    uint64_t frequency = CurrentFrequency();
    int core = CurrentCore();
    if (frequency && core >= 0)
        std::cout << std::fixed << std::setprecision(1) << "Current CPU core " << core << " frequency: " << double(CurrentFrequency()) / double(1000000000) << " GHz." << std::endl;
}

