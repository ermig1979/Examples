#include "defs.h"

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

void init(Buf& buf)
{
    for (int i = 0; i < buf.n; ++i)
        buf.p[i] = float(rand()) / float(RAND_MAX) - 0.5f;
}

inline float square(float x)
{
    return x * x;
}

inline float invalid(float a, float b, float e2)
{
    float d2 = square(a - b);
    return d2 > e2 && d2 > e2 * (a * a + b * b);
}

bool check(const Buf& control, const Buf& current, const std::string& desc, float eps = 0.001f)
{
    assert(control.n == current.n);
    float e2 = square(eps);
    for (int i = 0; i < control.n; ++i)
    {
        if (invalid(control.p[i], current.p[i], e2))
        {
            std::cout << desc << " : check error at " << i << ": ";
            std::cout << std::setprecision(4) << std::fixed << control.p[i] << " != " << current.p[i] << std::endl;
            return false;
        }
    }
    return true;
}


int main(int argc, char* argv[])
{

}