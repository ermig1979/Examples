#include "gemm.h"
#include "diff.h"
#include "time.h"
#include "amx.h"

namespace Amx
{
    static uint64_t PerfBf16L0(int count, const float * src, float * dst)
    {
        _tile_loadd(0, src + 0 * 256, 64);
        _tile_loadd(1, src + 1 * 256, 64);
        _tile_loadd(2, src + 2 * 256, 64);
        _tile_loadd(3, src + 3 * 256, 64);
        _tile_loadd(4, src + 4 * 256, 64);
        _tile_loadd(5, src + 5 * 256, 64);
        _tile_loadd(6, src + 6 * 256, 64);
        _tile_loadd(7, src + 7 * 256, 64);

        for(int i = 0; i < count; ++i)
        {
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);

            _tile_dpbf16ps(0, 5, 6);
            _tile_dpbf16ps(1, 5, 7);
            _tile_dpbf16ps(2, 4, 6);
            _tile_dpbf16ps(3, 4, 7);

            _tile_dpbf16ps(0, 5, 7);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 4, 6);

            _tile_dpbf16ps(0, 4, 7);
            _tile_dpbf16ps(1, 5, 6);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 6);
        }

        _tile_stored(0, dst + 0 * 256, 64);
        _tile_stored(1, dst + 1 * 256, 64);
        _tile_stored(2, dst + 2 * 256, 64);
        _tile_stored(3, dst + 3 * 256, 64);
        _tile_stored(4, dst + 4 * 256, 64);
        _tile_stored(5, dst + 5 * 256, 64);
        _tile_stored(6, dst + 6 * 256, 64);
        _tile_stored(7, dst + 7 * 256, 64);

        return count * 16 * uint64_t(16 * 1024);
    }

    void TestPerfBf16L0(double time)
    {
        std::cout << "Test L0 AMX BF16 performance: " << std::setprecision(3) << std::fixed;
        Mat32f stub(8, 16 * 16);
        Fill(stub);

        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfBf16L0(1024, stub.p, stub.p);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        double optime = t / n * 1000000000.0f * uint64_t(16 * 1024);
        std::cout << gflops << " GFLOPS; AMX BF16 time: " << optime << " nsec." << std::endl;
        if (stub.p[0])
            std::cout << " ";
    }

    //-------------------------------------------------------------------------------------------------

    static uint64_t PerfBf16L0v2(int count, const float* src, float* dst)
    {
        _tile_loadd(0, src + 0 * 256, 64);
        _tile_loadd(1, src + 1 * 256, 64);
        _tile_loadd(2, src + 2 * 256, 64);
        _tile_loadd(3, src + 3 * 256, 64);
        _tile_loadd(4, src + 4 * 256, 64);
        _tile_loadd(5, src + 5 * 256, 64);
        _tile_loadd(6, src + 6 * 256, 64);
        _tile_loadd(7, src + 7 * 256, 64);

        for (int i = 0; i < count; ++i)
        {
            _tile_dpbf16ps(0, 1, 5);
            _tile_dpbf16ps(0, 2, 7);
            _tile_dpbf16ps(0, 3, 6);
            _tile_dpbf16ps(0, 4, 7);

            _tile_dpbf16ps(0, 1,5);
            _tile_dpbf16ps(0, 2, 7);
            _tile_dpbf16ps(0, 3, 6);
            _tile_dpbf16ps(0, 4, 7);

            _tile_dpbf16ps(0, 1, 5);
            _tile_dpbf16ps(0, 2, 7);
            _tile_dpbf16ps(0, 3, 6);
            _tile_dpbf16ps(0, 4, 6);

            _tile_dpbf16ps(0, 1, 5);
            _tile_dpbf16ps(0, 2, 6);
            _tile_dpbf16ps(0, 3, 7);
            _tile_dpbf16ps(0, 4, 6);
        }

        _tile_stored(0, dst + 0 * 256, 64);
        _tile_stored(1, dst + 1 * 256, 64);
        _tile_stored(2, dst + 2 * 256, 64);
        _tile_stored(3, dst + 3 * 256, 64);
        _tile_stored(4, dst + 4 * 256, 64);
        _tile_stored(5, dst + 5 * 256, 64);
        _tile_stored(6, dst + 6 * 256, 64);
        _tile_stored(7, dst + 7 * 256, 64);

        return count * 16 * uint64_t(16 * 1024);
    }

    void TestPerfBf16L0v2(double time)
    {
        std::cout << "Test L0 AMX BF16 performance: " << std::setprecision(3) << std::fixed;
        Mat32f stub(8, 16 * 16);
        Fill(stub);

        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfBf16L0v2(1024, stub.p, stub.p);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        double optime = t / n * 1000000000.0f * uint64_t(16 * 1024);
        std::cout << gflops << " GFLOPS; AMX BF16 time: " << optime << " nsec (v2)." << std::endl;
        if (stub.p[0])
            std::cout << " ";
    }

    //-------------------------------------------------------------------------------------------------

    static uint64_t PerfInt8L0(int count, const int32_t* src, int32_t* dst)
    {
        _tile_loadd(0, src + 0 * 256, 64);
        _tile_loadd(1, src + 1 * 256, 64);
        _tile_loadd(2, src + 2 * 256, 64);
        _tile_loadd(3, src + 3 * 256, 64);
        _tile_loadd(4, src + 4 * 256, 64);
        _tile_loadd(5, src + 5 * 256, 64);
        _tile_loadd(6, src + 6 * 256, 64);
        _tile_loadd(7, src + 7 * 256, 64);

        for (int i = 0; i < count; ++i)
        {
            _tile_dpbuud(0, 4, 6);
            _tile_dpbuud(1, 4, 7);
            _tile_dpbuud(2, 5, 6);
            _tile_dpbuud(3, 5, 7);

            _tile_dpbuud(0, 5, 6);
            _tile_dpbuud(1, 5, 7);
            _tile_dpbuud(2, 4, 6);
            _tile_dpbuud(3, 4, 7);

            _tile_dpbuud(0, 5, 7);
            _tile_dpbuud(1, 4, 7);
            _tile_dpbuud(2, 5, 6);
            _tile_dpbuud(3, 4, 6);

            _tile_dpbuud(0, 4, 7);
            _tile_dpbuud(1, 5, 6);
            _tile_dpbuud(2, 4, 7);
            _tile_dpbuud(3, 5, 6);
        }

        _tile_stored(0, dst + 0 * 256, 64);
        _tile_stored(1, dst + 1 * 256, 64);
        _tile_stored(2, dst + 2 * 256, 64);
        _tile_stored(3, dst + 3 * 256, 64);
        _tile_stored(4, dst + 4 * 256, 64);
        _tile_stored(5, dst + 5 * 256, 64);
        _tile_stored(6, dst + 6 * 256, 64);
        _tile_stored(7, dst + 7 * 256, 64);

        return count * 16 * uint64_t(32 * 1024);
    }

    void TestPerfInt8L0(double time)
    {
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "Test L0 AMX INT8 performance: ";

        Mat32i stub(8, 16 * 16);
        Fill(stub);

        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfInt8L0(1024, stub.p, stub.p);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        double optime = t / n * 1000000000.0f * uint64_t(32 * 1024);
        std::cout << std::setprecision(3) << std::fixed << gflops << " GFLOPS; AMX INT8 time: " << optime << " nsec." << std::endl;
        if (stub.p[0])
            std::cout << " ";
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfBf16L1(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        if (zero)
        {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
        }
        else
        {
            _tile_stream_loadd(0, C + 0, ldc * 4);
            _tile_stream_loadd(1, C + 16, ldc * 4);
            _tile_stream_loadd(2, C + 16 * ldc + 0, ldc * 4);
            _tile_stream_loadd(3, C + 16 * ldc + 16, ldc * 4);
        }
        for (size_t k = 0; k < K; k += 32)
        {
            _tile_loadd(4, A + k, lda * 2);
            _tile_loadd(6, B + k * 32 + 0, 128);
            _tile_dpbf16ps(0, 4, 6);
            _tile_loadd(7, B + k * 32 + 32, 128);
            _tile_dpbf16ps(1, 4, 7);
            _tile_loadd(5, A + k + lda * 16, lda * 2);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }
        _tile_stored(0, C + 0, ldc * 4);
        _tile_stored(1, C + 16, ldc * 4);
        _tile_stored(2, C + 16 * ldc + 0, ldc * 4);
        _tile_stored(3, C + 16 * ldc + 16, ldc * 4);

        return uint64_t(K * 2 * 32 * 32);
    }

    void TestPerfBf16L1(double time)
    {
        std::cout << "Test L1 AMX BF16 performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

        const int L1 = (48 - 4) * 1024;
        const int K = L1 / 2 / 32 / 2;

        Mat16b a(32, K), b(K, 32);
        Mat32f c(32, 32);
        Fill(a); Fill(b); Fill(c);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfBf16L1(K, a.p, K, b.p, c.p, 32, n == 0 ? 1 : 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS." << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfMicroBf16L2(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        if (zero)
        {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
        }
        else
        {
            _tile_stream_loadd(0, C + 0, ldc * 4);
            _tile_stream_loadd(1, C + 16, ldc * 4);
            _tile_stream_loadd(2, C + 16 * ldc + 0, ldc * 4);
            _tile_stream_loadd(3, C + 16 * ldc + 16, ldc * 4);
        }
        for (size_t k = 0; k < K; k += 32)
        {
            _tile_stream_loadd(4, A + k, lda * 2);
            _tile_loadd(6, B + k * 32 + 0, 128);
            _tile_dpbf16ps(0, 4, 6);
            _tile_loadd(7, B + k * 32 + 32, 128);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stream_loadd(5, A + k + lda * 16, lda * 2);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }
        _tile_stored(0, C + 0, ldc * 4);
        _tile_stored(1, C + 16, ldc * 4);
        _tile_stored(2, C + 16 * ldc + 0, ldc * 4);
        _tile_stored(3, C + 16 * ldc + 16, ldc * 4);

        return uint64_t(K * 2 * 32 * 32);
    }

    inline uint64_t PerfMacroBf16L2(int M, int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        uint64_t n = 0;
        for (int i = 0; i < M; i += 32)
            n += PerfMicroBf16L2(K, A + i * lda, lda, B, C + i * ldc, ldc, zero);
        return n;
    }

    void TestPerfBf16L2(double time)
    {
        std::cout << "Test L2 AMX BF16 performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

        const int L1 = 48 * 1024, L2 = 2 * 1024 * 1024;
        const int K = L1 / 2 / 32;
        const int M = L2 / 2 / K / 2 / 32 * 32;

        Mat16b a(M, K), b(K, 32);
        Mat32f c(M, 32);
        Fill(a); Fill(b); Fill(c);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfMacroBf16L2(M, K, a.p, K, b.p, c.p, 32, n == 0 ? 1 : 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS." << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfMacroBf16L3(int M, int N, int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        uint64_t n = 0;
        for (int j = 0; j < N; j += 32)
        {
            for (int i = 0; i < M; i += 32)
                n += PerfMicroBf16L2(K, A + i * lda, lda, B + K * j, C + i * ldc + j, ldc, zero);
        }
        return n;
    }

    void TestPerfBf16L3(double time)
    {
        std::cout << "Test L3 AMX BF16 performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

        const int L1 = 48 * 1024, L2 = 2 * 1024 * 1024, L3 = 2 * 1024 * 1024;
        const int K = L1 / 2 / 32;
        const int M = L2 / 2 / K / 2 / 32 * 32;
        const int N = L3 / 2 / K / 32 * 32;

        Mat16b a(M, K), b(K, N);
        Mat32f c(M, N);
        Fill(a); Fill(b); Fill(c);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfMacroBf16L3(M, N, K, a.p, K, b.p, c.p, N, n == 0 ? 1 : 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS. " << std::endl;
    }

    //-------------------------------------------------------------------------------------------------


    void TestPerf(double time)
    {
        TestPerfBf16L0(time);

        TestPerfBf16L0v2(time);

        TestPerfInt8L0(time);

        TestPerfBf16L1(time);

        TestPerfBf16L2(time);
        
        TestPerfBf16L3(time);
    }
}


