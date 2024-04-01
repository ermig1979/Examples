
#include "mat.h"
#include "diff.h"
#include "time.h"
#include "gemm.h"

void Test32f(int M, int N, int K, const std::string& desc, Gemm32fPtr gemm, Gemm32fPtr control, double time = 1.0)
{
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "TEST " << desc << " :" << std::endl;

    Mat32f a(M, K), b(K, N), c0(M, N), c1(M, N);
    srand(0);
    Init(a, -1.0, 1.0, 1);
    Init(b, -1.0, 1.0, 1);

    Gemm32f(a, b, c0, control);
    double t = 0;
    int n = 0;
    while (t < time)
    {
        double start = Time();
        Gemm32f(a, b, c1, gemm);
        t += Time() - start;
        n++;
    }
    double gflops = 2 * double(M * N) * K * n / t / (1024 * 1024 * 1024);
    std::cout << desc << " : " << std::setprecision(3) << std::fixed << gflops << " GFLOPS; t = " << t / n * 1000.0f << " msec." << std::endl;

    Diff d;
    GetDiff(c0, c1, d);
    std::cout << " Diff: " << d.d.Info(6) << std::endl;
    std::cout << std::endl;
}

#define TEST32F(M, N, K, gemm, control) Test32f(M, N, K, #gemm, gemm, control)

int main(int argc, char* argv[])
{
    const int S = 1536;
    int M = S, N = S, K = S;
    if (argc > 1) M = N = K = atoi(argv[1]);
    if (argc > 2) N = K = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    //TEST32F(M, N, K, Base::Gemm32f, Avx512bw::Gemm32f);

    TEST32F(M, N, K, Avx2::Gemm32f, Avx512bw::Gemm32f);

    TEST32F(M, N, K, Avx512bw::Gemm32f, Avx512bw::Gemm32f);

    return 0;
}