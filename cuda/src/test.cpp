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

void init(cpu_buf_t & buf)
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

bool check(const cpu_buf_t & control, const cpu_buf_t & current, const std::string & desc, float eps = 0.001f)
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

void copy(const cpu_buf_t & src, gpu_buf_t & dst)
{
    assert(src.n == dst.n);
    cudaError_t error = cudaMemcpy(dst.p, src.p, src.n*sizeof(float), cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);
}

void copy(const gpu_buf_t & src, cpu_buf_t & dst)
{
    assert(src.n == dst.n);
    cudaError_t error = cudaMemcpy(dst.p, src.p, src.n*sizeof(float), cudaMemcpyDeviceToHost);
    assert(error == cudaSuccess);
}

struct opt_t
{
    int M, N, K, L;
    float T;

    opt_t(int argc, char* argv[])
    {
        const int S = 1152;
        M = S;
        N = S;
        K = S;
        L = 0;
        T = 1000.0f;
        if (argc > 1) M = atoi(argv[1]);
        if (argc > 2) N = atoi(argv[2]);
        if (argc > 3) K = atoi(argv[3]);
        if (argc > 4) L = atoi(argv[4]);
    }
};

bool test(gemm_t gemm, const std::string & desc, const opt_t & o,
        const cpu_buf_t & a, const cpu_buf_t & b, const cpu_buf_t & control)
{
    gpu_buf_t _a(o.M*o.K), _b(o.K*o.N), _c(o.M*o.N);

    copy(a, _a);
    copy(b, _b);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    assert(cudaEventCreate(&start) == cudaSuccess);
    assert(cudaEventCreate(&stop) == cudaSuccess);

    float t = 0, ms = 0;
    int n = 0;
    while(t < o.T)
    {
        assert(cudaEventRecord(start, NULL) == cudaSuccess);
        gemm(o.M, o.N, o.K, _a.p, _b.p, _c.p);
        assert(cudaEventRecord(stop, NULL) == cudaSuccess);
        assert(cudaEventSynchronize(stop) == cudaSuccess);
        assert(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess);
        t += ms;
        n++;
        std::cout << std::setprecision(1) << std::fixed << t/o.T*100.0f << "%\r" << std::flush;
    }
    double gflops = 2*double(o.M*o.N)*o.K*n / (t / 1000.0f) / (1000 * 1000 * 1000);

    cpu_buf_t current(o.M*o.N);
    copy(_c, current);
    cudaDeviceSynchronize();

    std::cout << desc << " : " << std::setprecision(3) << std::fixed << gflops << " GFLOPS; t = " << t/n << " msec." << std::endl<< std::flush;

    return check(control, current, desc);
}

void gemm_cpu(int M, int N, int K, const float * A, const float * B, float * C)
{
    for (int i = 0; i < M; ++i)
    {
        float * c = C + i * N;
        for (int j = 0; j < N; ++j)
            c[j] = 0;
        for (int k = 0; k < K; ++k)
        {
            const float * b = B + k * N;
            float a = A[i*K + k];
            for (int j = 0; j < N; ++j)
                c[j] += a * b[j];
        }
    }
}

int main(int argc, char* argv[])
{
    opt_t o(argc, argv);
    cpu_buf_t a(o.M*o.K), b(o.K*o.N), c(o.M*o.N);
    init(a);
    init(b);
    gemm_cpu(o.M, o.N, o.K, a.p, b.p, c.p);

    if (o.L <= 0 && !test(gemm_gpu_v0, "gemm_gpu_v0", o, a, b, c)) return 1;
}