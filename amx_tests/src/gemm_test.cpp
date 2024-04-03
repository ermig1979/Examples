#include "gemm.h"
#include "diff.h"
#include "time.h"

inline std::string ToStr(std::string value, int width)
{
    std::stringstream ss;
    ss << std::setfill(' ') << std::setw(width) << value;
    return ss.str();
}

void TestGemm32f(int M, int N, int K, const std::string& desc, Gemm32fPtr gemm, Gemm32fPtr control, double time = 1.0)
{
    std::cout << "Test " << desc << " : ";

    Mat32f a(M, K), b(K, N), c0(M, N), c1(M, N);
    srand(0);
    Init(a, -0.1, 0.1, 1);
    Init(b, -0.1, 0.1, 1);
    Fill(c0);
    Fill(c1);

    control(a.m, b.n, a.n, a.p, b.p, c0.p);
    double t = 0;
    int n = 0;
    while (t < time)
    {
        double start = Time();
        gemm(a.m, b.n, a.n, a.p, b.p, c1.p);;
        t += Time() - start;
        n++;
    }
    double gflops = 2 * double(M * N) * K * n / t / (1024 * 1024 * 1024);
    Diff d;
    GetDiff(c0, c1, d);
    std::cout << std::setprecision(3) << std::fixed << gflops << " GFLOPS; e = " << d.d.Abs() << std::endl;
}

#define TEST_GEMM32F(M, N, K, gemm, control) TestGemm32f(M, N, K, ToStr(#gemm, 20), gemm, control)

//-------------------------------------------------------------------------------------------------

void TestGemm32f16b(int M, int N, int K, const std::string& desc, Gemm32f16bPtr gemm, Gemm32fPtr control, double time = 1.0)
{
    std::cout << "Test " << desc << " : ";

    Mat32f a(M, K), b(K, N), c0(M, N), c1(M, N);
    srand(0);
    Init(a, -0.1, 0.1, 1);
    Init(b, -0.1, 0.1, 1);
    Fill(c0);
    Fill(c1);
    Mat16b _b(K, N);
    Amx::ReorderB(N, K, 32, b.p, _b.p);

    control(a.m, b.n, a.n, a.p, b.p, c0.p);
    double t = 0;
    int n = 0;
    while (t < time)
    {
        double start = Time();
        gemm(a.m, b.n, a.n, a.p, _b.p, c1.p);;
        t += Time() - start;
        n++;
    }
    double gflops = 2 * double(M * N) * K * n / t / (1024 * 1024 * 1024);
    Diff d;
    GetDiff(c0, c1, d);
    std::cout << std::setprecision(3) << std::fixed << gflops << " GFLOPS; e = " << d.d.Abs() << std::endl;
}

#define TEST_GEMM32F16B(M, N, K, gemm, control) TestGemm32f16b(M, N, K, ToStr(#gemm, 20), gemm, control)

//-------------------------------------------------------------------------------------------------

void TestGemm16b(int M, int N, int K, const std::string& desc, Gemm16bPtr gemm, Gemm32fPtr control, double time = 1.0)
{
    std::cout << "Test " << desc << " : ";

    Mat32f a(M, K), b(K, N), c0(M, N), c1(M, N);
    srand(0);
    Init(a, -0.1, 0.1, 1);
    Init(b, -0.1, 0.1, 1);
    Fill(c0);
    Fill(c1);
    Mat16b _a(M, K);
    Amx::ConvertA(M, K, a.p, _a.p);
    Mat16b _b(K, N);
    Amx::ReorderB(N, K, 32, b.p, _b.p);

    control(a.m, b.n, a.n, a.p, b.p, c0.p);
    double t = 0;
    int n = 0;
    while (t < time)
    {
        double start = Time();
        gemm(a.m, b.n, a.n, _a.p, _b.p, c1.p);;
        t += Time() - start;
        n++;
    }
    double gflops = 2 * double(M * N) * K * n / t / (1024 * 1024 * 1024);
    Diff d;
    GetDiff(c0, c1, d);
    std::cout << std::setprecision(3) << std::fixed << gflops << " GFLOPS; e = " << d.d.Abs() << std::endl;
}

#define TEST_GEMM16B(M, N, K, gemm, control) TestGemm16b(M, N, K, ToStr(#gemm, 20), gemm, control)

//-------------------------------------------------------------------------------------------------

bool TestGemm(int M, int N, int K)
{
    //TEST_GEMM32F(M, N, K, Base::Gemm32f, Avx512bw::Gemm32f);

    //TEST_GEMM32F(M, N, K, Avx2::Gemm32f, Avx512bw::Gemm32f);

    TEST_GEMM32F(M, N, K, Avx512bw::Gemm32f, Avx512bw::Gemm32f);

    //TEST_GEMM32F(M, N, K, Base::Gemm16b, Avx512bw::Gemm32f);

    TEST_GEMM32F(M, N, K, Amx::Gemm32f, Avx512bw::Gemm32f);

    TEST_GEMM32F16B(M, N, K, Amx::Gemm32f16b, Avx512bw::Gemm32f);

    TEST_GEMM32F16B(M, N, K, Amx::Gemm32f16bV2, Avx512bw::Gemm32f);

    TEST_GEMM16B(M, N, K, Amx::Gemm16b, Avx512bw::Gemm32f);

    TEST_GEMM32F(M, N, K, Amx::StubMicro16b, Amx::StubMicro16b);

    TEST_GEMM32F(M, N, K, Amx::StubMacro16b, Amx::StubMacro16b);

    return true;
}

