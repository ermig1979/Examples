#pragma once

#include "Defs.h"

namespace Test
{
    inline double Time()
    {
        using namespace std::chrono;
        const static time_point<high_resolution_clock> start = high_resolution_clock::now();
        time_point<high_resolution_clock> time = high_resolution_clock::now();
        return duration<double>(time - start).count();
    }

    inline void Sleep(unsigned int miliseconds)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
    }

    inline String PrettyString(long long val)
    {
        String str = std::to_string(val);
        ptrdiff_t n = str.length() - 3;
        ptrdiff_t end = (val >= 0) ? 0 : 1;
        while (n > end) 
        {
            str.insert(n, "`");
            n -= 3;
        }
        return str;
    }
}
