#include "defs.h"

struct cpu_buf_t
{
    float* p;
    int n;

    cpu_buf_t(int size)
        : n(size)
        , p((float*)_mm_malloc(size * 4, 64))
    {
    }

    ~cpu_buf_t()
    {
        _mm_free(p);
    }
};

void copy(const cpu_buf_t& src, gpu_buf_t& dst)
{
    assert(src.n == dst.n);
    cudaError_t error = cudaMemcpy(dst.p, src.p, src.n * sizeof(float), cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);
}

void copy(const gpu_buf_t& src, cpu_buf_t& dst)
{
    assert(src.n == dst.n);
    cudaError_t error = cudaMemcpy(dst.p, src.p, src.n * sizeof(float), cudaMemcpyDeviceToHost);
    assert(error == cudaSuccess);
}

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

struct opt_t
{
    int M, N, K, L;
    float T;

    opt_t(int argc, char* argv[])
    {
        const int S = 1024;
        M = S;
        N = S;
        K = S;
        L = 0;
        T = 1000.0f;
        if (argc == 2) M = atoi(argv[1]), N = atoi(argv[1]), K = atoi(argv[1]);
        else
        {
            if (argc > 1) M = atoi(argv[1]);
            if (argc > 2) N = atoi(argv[2]);
            if (argc > 3) K = atoi(argv[3]);
            if (argc > 4) L = atoi(argv[4]);
        }
    }
};

void control(gemm_t gemm, const opt_t& o, const cpu_buf_t& a, const cpu_buf_t& b, cpu_buf_t & c)
{
    gpu_buf_t _a(o.M * o.K), _b(o.K * o.N), _c(o.M * o.N);
    copy(a, _a);
    copy(b, _b);
    gemm(o.M, o.N, o.K, _a.p, _b.p, _c.p);
    copy(_c, c);
}

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
        n += gemm(o.M, o.N, o.K, _a.p, _b.p, _c.p);
        assert(cudaEventRecord(stop, NULL) == cudaSuccess);
        assert(cudaEventSynchronize(stop) == cudaSuccess);
        assert(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess);
        t += ms;
        std::cout << std::setprecision(1) << std::fixed << t/o.T*100.0f << "%\r" << std::flush;
    }
    assert(cudaEventDestroy(start) == cudaSuccess);
    assert(cudaEventDestroy(stop) == cudaSuccess);

    double gflops = 2*double(o.M*o.N)*o.K*n / (t / 1000.0f) / (1000 * 1000 * 1000);

    cpu_buf_t current(o.M*o.N);
    copy(_c, current);
    cudaDeviceSynchronize();

    std::cout << desc << " : " << std::setprecision(3) << std::fixed << gflops << " GFLOPS; t = " << t/n << " msec." << std::endl<< std::flush;

    return check(control, current, desc);
}

int gemm_cpu(int M, int N, int K, const float * A, const float * B, float * C)
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
    return 1;
}

void print_info(const opt_t & o)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
    }
}

int main(int argc, char* argv[])
{
    opt_t o(argc, argv);
    print_info(o);

    cpu_buf_t a(o.M*o.K), b(o.K*o.N), c(o.M*o.N);
    init(a);
    init(b);
    control(gemm_cublas, o, a, b, c);

    if (o.L <= 0 && !test(gemm_cublas, " gemm_cublas", o, a, b, c)) return 1;
    if (o.L <= 0 && !test(gemm_gpu_v0a, "gemm_gpu_v0a", o, a, b, c)) return 1;
    if (o.L <= 1 && !test(gemm_gpu_v1a, "gemm_gpu_v1a", o, a, b, c)) return 1;
    if (o.L <= 2 && !test(gemm_gpu_v2a, "gemm_gpu_v2a", o, a, b, c)) return 1;
    if (o.L <= 3 && !test(gemm_gpu_v3a, "gemm_gpu_v3a", o, a, b, c)) return 1;
    if (o.L <= 4 && !test(gemm_gpu_v4a, "gemm_gpu_v4a", o, a, b, c)) return 1;
    if (o.L <= 4 && !test(gemm_gpu_v4b, "gemm_gpu_v4b", o, a, b, c)) return 1;
    if (o.L <= 4 && !test(gemm_gpu_v4c, "gemm_gpu_v4c", o, a, b, c)) return 1;
    if (o.L <= 5 && !test(gemm_gpu_v5a, "gemm_gpu_v5a", o, a, b, c)) return 1;
    if (o.L <= 5 && !test(gemm_gpu_v5b, "gemm_gpu_v5b", o, a, b, c)) return 1;
    if (o.L <= 5 && !test(gemm_gpu_v5c, "gemm_gpu_v5c", o, a, b, c)) return 1;
}