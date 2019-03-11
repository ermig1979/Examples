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

void init(buf_t & buf)
{
    for (int i = 0; i < buf.n; ++i)
        buf.p[i] = float(rand()) / float(RAND_MAX) - 0.5f;
}

inline float square(float x)
{
    return x*x;
}

inline float invalid(float a, float b, float e2)
{
    float d2 = square(a - b);
    return d2 > e2 && d2 > e2*(a*a + b*b);
}

bool check(const buf_t & control, const buf_t & current, const std::string & desc, float eps = 0.001f)
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

int S = 1152;
int M = S, N = S, K = S, L = 0;
double TIME = 1.0;

bool test(gemm_t gemm, const std::string & desc, const buf_t & a, const buf_t & b, const buf_t & control)
{
    buf_t current(M*N);

    double t = 0;
    int n = 0;
    while(t < TIME)
    {
        double start = time();
        gemm(M, N, K, a.p, b.p, current.p);
        t += time() - start;
        n++;
    }
    double gflops = 2*double(M*N)*K*n / t / (1024* 1024* 1024);

    std::cout << desc << " : " << std::setprecision(3) << std::fixed << gflops << " GFLOPS; t = " << t/n*1000.0f << " msec." << std::endl;

    return check(control, current, desc);
}

int main(int argc, char* argv[])
{
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) L = atoi(argv[4]);
    buf_t a(M*K), b(K*N), c(M*N);
    init(a);
    init(b);
    gemm_v4(M, N, K, a.p, b.p, c.p);

    if (L <= 0 && !test(gemm_v0, "gemm_v0", a, b, c)) return 1;
    if (L <= 1 && !test(gemm_v1, "gemm_v1", a, b, c)) return 1;
    if (L <= 2 && !test(gemm_v2, "gemm_v2", a, b, c)) return 1;
    if (L <= 3 && !test(gemm_v3, "gemm_v3", a, b, c)) return 1;
    if (L <= 4 && !test(gemm_v4, "gemm_v4", a, b, c)) return 1;
    if (L <= 5 && !test(gemm_v5, "gemm_v5", a, b, c)) return 1;
    if (L <= 6 && !test(gemm_v6, "gemm_v6", a, b, c)) return 1;
    if (L <= 7 && !test(gemm_v7, "gemm_v7", a, b, c)) return 1;

    return 0;
}