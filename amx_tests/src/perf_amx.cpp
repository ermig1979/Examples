#include "gemm.h"
#include "diff.h"
#include "time.h"
#include "amx.h"

namespace Amx
{
    static uint64_t PerfBf16L0_2x2(int count, int step, const float * src, float * dst)
    {
        _tile_loadd(0, src + 0 * 256, 64);
        _tile_loadd(1, src + 1 * 256, 64);
        _tile_loadd(2, src + 2 * 256, 64);
        _tile_loadd(3, src + 3 * 256, 64);
        _tile_loadd(4, src + 4 * 256, 64);
        _tile_loadd(5, src + 5 * 256, 64);
        _tile_loadd(6, src + 6 * 256, 64);
        _tile_loadd(7, src + 7 * 256, 64);

        for(int i = 0, s = step * 2; i < count; i += s)
        {
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }

        _tile_stored(0, dst + 0 * 256, 64);
        _tile_stored(1, dst + 1 * 256, 64);
        _tile_stored(2, dst + 2 * 256, 64);
        _tile_stored(3, dst + 3 * 256, 64);
        _tile_stored(4, dst + 4 * 256, 64);
        _tile_stored(5, dst + 5 * 256, 64);
        _tile_stored(6, dst + 6 * 256, 64);
        _tile_stored(7, dst + 7 * 256, 64);

        return count * uint64_t(2 * 32 * 32);
    }

    void TestPerfBf16L0_2x2(double time, int step = 16)
    {
        std::cout << "Test L0 AMX BF16 2x2x" << step * 2 << " performance: " << std::setprecision(3) << std::fixed;
        Mat32f stub(8, 16 * 16);
        Fill(stub);

        TileConf conf;
        conf.SetMax();
        conf.colsb[4] = uint16_t(step * 4);
        conf.colsb[5] = uint16_t(step * 4);
        conf.rows[6] = uint8_t(step);
        conf.rows[7] = uint8_t(step);
        _tile_loadconfig(&conf);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfBf16L0_2x2(2048, step, stub.p, stub.p);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        double optime = t / n * 1000000000.0f * uint64_t(step * 1024);
        std::cout << gflops << " GFLOPS; AMX BF16 optime: " << optime << " nsec." << std::endl;
        if (stub.p[0])
            std::cout << " ";
    }

    //-------------------------------------------------------------------------------------------------

    static uint64_t PerfBf16L0_1x1(int count, const float* src, float* dst)
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

    void TestPerfBf16L0_1x1(double time)
    {
        std::cout << "Test L0 AMX BF16 1x1 performance: " << std::setprecision(3) << std::fixed;
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
            n += PerfBf16L0_1x1(1024, stub.p, stub.p);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        double optime = t / n * 1000000000.0f * uint64_t(16 * 1024);
        std::cout << gflops << " GFLOPS; AMX BF16 optime: " << optime << " nsec." << std::endl;
        if (stub.p[0])
            std::cout << " ";
    }

    //-------------------------------------------------------------------------------------------------

    static uint64_t PerfInt8L0_2x2(int count, const int32_t* src, int32_t* dst)
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

    void TestPerfInt8L0_2x2(double time)
    {
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "Test L0 AMX INT8 2x2 performance: ";

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
            n += PerfInt8L0_2x2(1024, stub.p, stub.p);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        double optime = t / n * 1000000000.0f * uint64_t(32 * 1024);
        std::cout << std::setprecision(3) << std::fixed << gflops << " GFLOPS; AMX INT8 optime: " << optime << " nsec." << std::endl;
        if (stub.p[0])
            std::cout << " ";
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfBf16L1_2x2(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
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

    void TestPerfBf16L1_2x2(double time)
    {
        std::cout << "Test L1 AMX BF16 2x2 performance: " << std::setprecision(3) << std::fixed;

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
            n += PerfBf16L1_2x2(K, a.p, K, b.p, c.p, 32, n == 0 ? 1 : 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS." << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfBf16L1_2x1(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        if (zero)
        {
            _tile_zero(0);
            _tile_zero(2);
        }
        else
        {
            _tile_stream_loadd(0, C + 0, ldc * 4);
            _tile_stream_loadd(2, C + 16 * ldc + 0, ldc * 4);
        }
        for (size_t k = 0; k < K; k += 32)
        {
            _tile_loadd(4, A + k, lda * 2);
            _tile_loadd(6, B + k * 16 + 0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_loadd(5, A + k + lda * 16, lda * 2);
            _tile_dpbf16ps(2, 5, 6);
        }
        _tile_stored(0, C + 0, ldc * 4);
        _tile_stored(2, C + 16 * ldc + 0, ldc * 4);

        return uint64_t(K * 2 * 32 * 16);
    }

    void TestPerfBf16L1_2x1(double time)
    {
        std::cout << "Test L1 AMX BF16 2x1 performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

        const int L1 = 48 * 1024;
        const int K = L1 / 2 / 48;

        Mat16b a(32, K), b(K, 16);
        Mat32f c(32, 16);
        Fill(a); Fill(b); Fill(c);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfBf16L1_2x1(K, a.p, K, b.p, c.p, 16, n == 0 ? 1 : 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS." << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfBf16L1_1x2(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        if (zero)
        {
            _tile_zero(0);
            _tile_zero(1);
        }
        else
        {
            _tile_stream_loadd(0, C + 0, ldc * 4);
            _tile_stream_loadd(1, C + 16, ldc * 4);
        }
        for (size_t k = 0; k < K; k += 32)
        {
            _tile_loadd(4, A + k, lda * 2);
            _tile_loadd(6, B + k * 32 + 0, 128);
            _tile_dpbf16ps(0, 4, 6);
            _tile_loadd(7, B + k * 32 + 32, 128);
            _tile_dpbf16ps(1, 4, 7);
        }
        _tile_stored(0, C + 0, ldc * 4);
        _tile_stored(1, C + 16, ldc * 4);

        return uint64_t(K * 2 * 16 * 32);
    }

    void TestPerfBf16L1_1x2(double time)
    {
        std::cout << "Test L1 AMX BF16 1x2 performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

        const int L1 = 48 * 1024;
        const int K = L1 / 2 / 48;

        Mat16b a(16, K), b(K, 32);
        Mat32f c(16, 32);
        Fill(a); Fill(b); Fill(c);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfBf16L1_1x2(K, a.p, K, b.p, c.p, 32, n == 0 ? 1 : 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS." << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfBf16L1_1x1(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        if (zero)
        {
            _tile_zero(0);
        }
        else
        {
            _tile_stream_loadd(0, C + 0, ldc * 4);
        }
        for (size_t k = 0; k < K; k += 32)
        {
            _tile_loadd(4, A + k, lda * 2);
            _tile_loadd(6, B + k * 16, 64);
            _tile_dpbf16ps(0, 4, 6);
        }
        _tile_stored(0, C + 0, ldc * 4);

        return uint64_t(K * 2 * 16 * 16);
    }

    void TestPerfBf16L1_1x1(double time)
    {
        std::cout << "Test L1 AMX BF16 1x1 performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

        const int L1 = (48 - 4) * 1024;
        const int K = L1 / 2 / 16 / 2;

        Mat16b a(16, K), b(K, 16);
        Mat32f c(16, 16);
        Fill(a); Fill(b); Fill(c);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfBf16L1_1x1(K, a.p, K, b.p, c.p, 16, n == 0 ? 1 : 0);
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

    inline uint64_t PerfMicroBf16L3_2x2(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
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

    inline uint64_t PerfMacroBf16L3_2x2(int M, int N, int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        uint64_t n = 0;
        for (int j = 0; j < N; j += 32)
        {
            for (int i = 0; i < M; i += 32)
                n += PerfMicroBf16L3_2x2(K, A + i * lda, lda, B + K * j, C + i * ldc + j, ldc, zero);
        }
        return n;
    }

    void TestPerfBf16L3_2x2(double time)
    {
        std::cout << "Test L3 AMX BF16 2x2 performance: " << std::setprecision(3) << std::fixed;

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
            n += PerfMacroBf16L3_2x2(M, N, K, a.p, K, b.p, c.p, N, n == 0 ? 1 : 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS. " << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfMicroBf16L3_2x1(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        if (zero)
        {
            _tile_zero(0);
            _tile_zero(2);
        }
        else
        {
            _tile_stream_loadd(0, C + 0, ldc * 4);
            _tile_stream_loadd(2, C + 16 * ldc + 0, ldc * 4);
        }
        for (size_t k = 0; k < K; k += 32)
        {
            _tile_stream_loadd(4, A + k, lda * 2);
            _tile_loadd(6, B + k * 16 + 0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stream_loadd(5, A + k + lda * 16, lda * 2);
            _tile_dpbf16ps(2, 5, 6);
        }
        _tile_stored(0, C + 0, ldc * 4);
        _tile_stored(2, C + 16 * ldc + 0, ldc * 4);

        return uint64_t(K * 2 * 32 * 16);
    }

    inline uint64_t PerfMacroBf16L3_2x1(int M, int N, int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        uint64_t n = 0;
        for (int j = 0; j < N; j += 16)
        {
            for (int i = 0; i < M; i += 32)
                n += PerfMicroBf16L3_2x1(K, A + i * lda, lda, B + K * j, C + i * ldc + j, ldc, zero);
        }
        return n;
    }

    void TestPerfBf16L3_2x1(double time)
    {
        std::cout << "Test L3 AMX BF16 2x1 performance: " << std::setprecision(3) << std::fixed;

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
            n += PerfMacroBf16L3_2x1(M, N, K, a.p, K, b.p, c.p, N, n == 0 ? 1 : 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS. " << std::endl;
    }


    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfMicroBf16L3_1x2(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        if (zero)
        {
            _tile_zero(0);
            _tile_zero(1);
        }
        else
        {
            _tile_stream_loadd(0, C + 0, ldc * 4);
            _tile_stream_loadd(1, C + 16, ldc * 4);
        }
        for (size_t k = 0; k < K; k += 32)
        {
            _tile_stream_loadd(4, A + k, lda * 2);
            _tile_loadd(6, B + k * 32 + 0, 128);
            _tile_dpbf16ps(0, 4, 6);
            _tile_loadd(7, B + k * 32 + 32, 128);
            _tile_dpbf16ps(1, 4, 7);
        }
        _tile_stored(0, C + 0, ldc * 4);
        _tile_stored(1, C + 16, ldc * 4);

        return uint64_t(K * 2 * 16 * 32);
    }

    inline uint64_t PerfMacroBf16L3_1x2(int M, int N, int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        uint64_t n = 0;
        for (int j = 0; j < N; j += 32)
        {
            for (int i = 0; i < M; i += 16)
                n += PerfMicroBf16L3_1x2(K, A + i * lda, lda, B + K * j, C + i * ldc + j, ldc, zero);
        }
        return n;
    }

    void TestPerfBf16L3_1x2(double time)
    {
        std::cout << "Test L3 AMX BF16 1x2 performance: " << std::setprecision(3) << std::fixed;

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
            n += PerfMacroBf16L3_1x2(M, N, K, a.p, K, b.p, c.p, N, n == 0 ? 1 : 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS. " << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    template<int X> inline uint64_t PerfMicroBf16L3_1xX(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        if (zero)
        {
            if (X > 0) _tile_zero(0);
            if (X > 1) _tile_zero(1);
            if (X > 2) _tile_zero(2);
            if (X > 3) _tile_zero(3);
            if (X > 4) _tile_zero(4);
            if (X > 5) _tile_zero(5);
        }
        else
        {
            if (X > 0) _tile_stream_loadd(0, C + 0 * 16, ldc * 4);
            if (X > 1) _tile_stream_loadd(1, C + 1 * 16, ldc * 4);
            if (X > 2) _tile_stream_loadd(2, C + 2 * 16, ldc * 4);
            if (X > 3) _tile_stream_loadd(3, C + 3 * 16, ldc * 4);
            if (X > 4) _tile_stream_loadd(4, C + 4 * 16, ldc * 4);
            if (X > 5) _tile_stream_loadd(5, C + 5 * 16, ldc * 4);
        }
        for (size_t k = 0; k < K; k += 32)
        {
            _tile_stream_loadd(6, A + k, lda * 2);
            if (X > 0) { _tile_loadd(7, B + k * 16 * X + 0 * 32 * 16, 64); _tile_dpbf16ps(0, 6, 7); }
            if (X > 1) { _tile_loadd(7, B + k * 16 * X + 1 * 32 * 16, 64); _tile_dpbf16ps(1, 6, 7); }
            if (X > 2) { _tile_loadd(7, B + k * 16 * X + 2 * 32 * 16, 64); _tile_dpbf16ps(2, 6, 7); }
            if (X > 3) { _tile_loadd(7, B + k * 16 * X + 3 * 32 * 16, 64); _tile_dpbf16ps(3, 6, 7); }
            if (X > 4) { _tile_loadd(7, B + k * 16 * X + 4 * 32 * 16, 64); _tile_dpbf16ps(4, 6, 7); }
            if (X > 5) { _tile_loadd(7, B + k * 16 * X + 5 * 32 * 16, 64); _tile_dpbf16ps(5, 6, 7); }
        }
        if (X > 0) _tile_stored(0, C + 0 * 16, ldc * 4);
        if (X > 1) _tile_stored(1, C + 1 * 16, ldc * 4);
        if (X > 2) _tile_stored(2, C + 2 * 16, ldc * 4);
        if (X > 3) _tile_stored(3, C + 3 * 16, ldc * 4);
        if (X > 4) _tile_stored(4, C + 4 * 16, ldc * 4);
        if (X > 5) _tile_stored(5, C + 5 * 16, ldc * 4);

        return uint64_t(K * 2 * 16 * 16 * X);
    }

    template<int X> inline uint64_t PerfMacroBf16L3_1xX(int M, int N, int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        uint64_t n = 0;
        for (int j = 0; j < N; j += 16 * X)
        {
            for (int i = 0; i < M; i += 16)
                n += PerfMicroBf16L3_1xX<X>(K, A + i * lda, lda, B + K * j, C + i * ldc + j, ldc, zero);
        }
        return n;
    }

    template<int X> void TestPerfBf16L3_1xX(double time)
    {
        std::cout << "Test L3 AMX BF16 1x" << X << " performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

        const int L1 = 48 * 1024, L2 = 2 * 1024 * 1024, L3 = 2 * 1024 * 1024;
        const int K = L1 / 2 / 32;
        const int M = L2 / 2 / K / 2 / 32 * 32;
        const int N = L3 / 2 / K / (16 * X) * (16 * X);

        Mat16b a(M, K), b(K, N);
        Mat32f c(M, N);
        Fill(a); Fill(b); Fill(c);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfMacroBf16L3_1xX<X>(M, N, K, a.p, K, b.p, c.p, N, n == 0 ? 1 : 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS. " << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    void TestPerf(double time)
    {
        TestPerfBf16L0_2x2(time, 16);
        TestPerfBf16L0_2x2(time, 8);
        TestPerfBf16L0_2x2(time, 1);
        TestPerfBf16L0_1x1(time);
        TestPerfInt8L0_2x2(time);

        TestPerfBf16L1_2x2(time);
        TestPerfBf16L1_2x1(time);
        TestPerfBf16L1_1x2(time);
        TestPerfBf16L1_1x1(time);

        TestPerfBf16L2(time);
        
        TestPerfBf16L3_2x2(time);
        TestPerfBf16L3_2x1(time);
        TestPerfBf16L3_1x2(time);

        TestPerfBf16L3_1xX<1>(time);
        TestPerfBf16L3_1xX<2>(time);
        TestPerfBf16L3_1xX<3>(time);
        TestPerfBf16L3_1xX<4>(time);
        TestPerfBf16L3_1xX<5>(time);
        TestPerfBf16L3_1xX<6>(time);
    }
}


