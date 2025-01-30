#include "test.h"
#include "amx.h"

namespace Amx
{
    static inline uint64_t PerfBf16L0(int count, int step)
    {
        for(int i = 0; i < count; i += step)
        {
#if 0
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(0, 4, 7);
            _tile_dpbf16ps(0, 5, 6);
            _tile_dpbf16ps(0, 5, 7);
#else
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
#endif
        }
        return count * uint64_t(2 * 16 * 16) * 4;
    }

    void TestPerfBf16L0(double time, int step)
    {
        std::cout << "Test L0 AMX BF16 16x16x" << step << " performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
        conf.colsb[4] = uint16_t(step * 2);
        conf.colsb[5] = uint16_t(step * 2);
        conf.rows[6] = uint8_t(step / 2);
        conf.rows[7] = uint8_t(step / 2);
        _tile_loadconfig(&conf);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfBf16L0(1024 * step, step);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        double optime = t / n * 1000000000.0f * uint64_t(step * 512);
        std::cout << gflops << " GFLOPS; optime: " << optime << " nsec." << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    static inline uint64_t PerfInt8L0(int count)
    {
        for (int i = 0; i < count; i += 4)
        {
            _tile_dpbuud(0, 4, 6);
            _tile_dpbuud(1, 4, 7);
            _tile_dpbuud(2, 5, 6);
            _tile_dpbuud(3, 5, 7);
        }
        return count * INT8_OPS;
    }

    void TestPerfInt8L0(double time)
    {
        std::cout << "Test L0 AMX INT8 16x16x64 performance: " << std::fixed << std::setprecision(3);

        TileConf conf;
        _tile_loadconfig(&conf);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfInt8L0(1024);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        double optime = t / n * 1000000000.0f * INT8_OPS;
        std::cout << gflops << " GFLOPS; optime: " << optime << " nsec." << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    template<int flags> inline uint64_t PerfBf16L1(int count, uint8_t* buf)
    {
        uint8_t* C = buf, * A0 = buf + 4 * 1024, * A1 = A0 + count * 1024, * B0 = A1 + count * 1024, * B1 = B0 + count * 1024;
        if (flags & 1)
        {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
        }
        if(flags & 2)
        {
            _tile_stream_loadd(0, C + 0 * 1024, 64);
            _tile_stream_loadd(1, C + 1 * 1024, 64);
            _tile_stream_loadd(2, C + 2 * 1024, 64);
            _tile_stream_loadd(3, C + 3 * 1024, 64);
        }
        if (flags & 4)
        {
            _tile_loadd(0, C + 0 * 1024, 64);
            _tile_loadd(1, C + 1 * 1024, 64);
            _tile_loadd(2, C + 2 * 1024, 64);
            _tile_loadd(3, C + 3 * 1024, 64);
        }
        if (flags & 16)
        {
            int i = 0, count1 = count - 1;
            _tile_stream_loadd(4, A0 + i * 1024, 64);
            _tile_loadd(6, B0 + i * 1024, 64);
            for (; i < count1; i++)
            {
                _tile_loadd(7, B1 + i * 1024, 64);
                _tile_stream_loadd(5, A1 + i * 1024, 64);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                _tile_stream_loadd(4, A0 + i * 1024 + 1024, 64);
                _tile_dpbf16ps(2, 5, 6);
                _tile_loadd(6, B0 + i * 1024 + 1024, 64);
                _tile_dpbf16ps(3, 5, 7);
            }
            {
                _tile_loadd(7, B1 + i * 1024, 64);
                _tile_stream_loadd(5, A1 + i * 1024, 64);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }        
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                _tile_loadd(4, A0 + i * 1024, 64);
                _tile_loadd(5, A1 + i * 1024, 64);
                _tile_loadd(6, B0 + i * 1024, 64);
                _tile_loadd(7, B1 + i * 1024, 64);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
        }
        if (flags & 8)
        {
            _tile_stored(0, C + 0 * 1024, 64);
            _tile_stored(1, C + 1 * 1024, 64);
            _tile_stored(2, C + 2 * 1024, 64);
            _tile_stored(3, C + 3 * 1024, 64);
        }

        return uint64_t(count * BF16_OPS * 4);
    }

    template<int flags> void TestPerfBf16L1(int count, double time)
    {
        std::cout << "Test L1 AMX BF16 2x2 performance: " << std::setprecision(3) << std::fixed;
        TileConf conf;
        _tile_loadconfig(&conf);

        Mat8u buf(1024 * 32, 1024 * 32);
        Fill(buf);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfBf16L1<flags>(count, buf.p);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS. f=" << flags << "; ";
        if (count * 4 >= 1024 * 1)
            std::cout << double(count * 4) / 1024 << " MB" << std::endl;
        else
            std::cout << count * 4 << " kB" << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    template<int N> inline uint64_t LoadLongRows(int count, uint8_t* buf)
    {
        uint8_t* A0 = buf, * A1 = A0 + count * 1024, * A2 = A1 + count * 1024, *A3 = A2 + count * 1024;
        for (int i = 0; i < count; i++)
        {
            if (N > 0) { _tile_loadd(0, A0 + i * 64, 64 * count); };
            if (N > 1) { _tile_loadd(1, A1 + i * 64, 64 * count); };
            if (N > 2) { _tile_loadd(2, A2 + i * 64, 64 * count); };
            if (N > 3) { _tile_loadd(3, A3 + i * 64, 64 * count); };
        }
        return uint64_t(count * 1024 * N);
    }

    template<int N> void TestLoadLongRows(int count, double time)
    {
        std::cout << "Test AMX loading (long rows): " << std::setprecision(3) << std::fixed;
        TileConf conf;
        _tile_loadconfig(&conf);

        Mat8u buf(1024, count * N);
        Fill(buf);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += LoadLongRows<N>(count, buf.p);
            t += Time() - start;
        }
        double gbps = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gbps << " GB/S. size =  ";
        if (count * N >= 1024 * 1)
            std::cout << double(count * N) / 1024 << " MB" << std::endl;
        else
            std::cout << count * N << " kB" << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    template<int N> inline uint64_t LoadCompact(int count, uint8_t* buf)
    {
        uint8_t* A0 = buf, * A1 = A0 + count * 1024, * A2 = A1 + count * 1024, * A3 = A2 + count * 1024;
        for (int i = 0; i < count; i++)
        {
            if (N > 0) { _tile_loadd(0, A0 + i * 1024, 64); };
            if (N > 1) { _tile_loadd(1, A1 + i * 1024, 64); };
            if (N > 2) { _tile_loadd(2, A2 + i * 1024, 64); };
            if (N > 3) { _tile_loadd(3, A3 + i * 1024, 64); };
        }
        return uint64_t(count * 1024 * N);
    }

    template<int N> void TestLoadCompact(int count, double time)
    {
        std::cout << "Test AMX loading (compact): " << std::setprecision(3) << std::fixed;
        TileConf conf;
        _tile_loadconfig(&conf);

        Mat8u buf(1024, count * N);
        Fill(buf);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += LoadCompact<N>(count, buf.p);
            t += Time() - start;
        }
        double gbps = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gbps << " GB/S. size =  ";
        if (count * N >= 1024 * 1)
            std::cout << double(count * N) / 1024 << " MB" << std::endl;
        else
            std::cout << count * N << " kB" << std::endl;
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
            //_tile_stream_loadd(0, C + 0, ldc * 4);
            //_tile_stream_loadd(1, C + 16, ldc * 4);
            //_tile_stream_loadd(2, C + 16 * ldc + 0, ldc * 4);
            //_tile_stream_loadd(3, C + 16 * ldc + 16, ldc * 4);
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
        //_tile_stored(0, C + 0, ldc * 4);
        //_tile_stored(1, C + 16, ldc * 4);
        //_tile_stored(2, C + 16 * ldc + 0, ldc * 4);
        //_tile_stored(3, C + 16 * ldc + 16, ldc * 4);

        return uint64_t(K * 2 * 32 * 32);
    }

    void TestPerfBf16L1_2x2(double time)
    {
        std::cout << "Test L1 AMX BF16 2x2 performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
        _tile_loadconfig(&conf);

        const int L1 = (48) * 1024;
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
            //_tile_stream_loadd(0, C + 0, ldc * 4);
        }
        for (size_t k = 0; k < K; k += 32)
        {
            _tile_loadd(4, A + k, lda * 2);
            _tile_loadd(6, B + k * 16, 64);
            _tile_dpbf16ps(0, 4, 6);
        }
        //_tile_stored(0, C + 0, ldc * 4);

        return uint64_t(K * 2 * 16 * 16);
    }

    void TestPerfBf16L1_1x1(double time)
    {
        std::cout << "Test L1 AMX BF16 1x1 performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
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
            n += PerfMacroBf16L2(M, K, a.p, K, b.p, c.p, 32, 1);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS." << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfMicroBf16L3(int K, const uint16_t* A0, const uint16_t* A1, const uint16_t* B0, const uint16_t* B1, float* C, int ldc, int zero)
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
#if 0
        _tile_stream_loadd(4, A0, 64);
        _tile_loadd(6, B0, 64);
        int k = 0, KB = K  - 32;
        for (; k < KB; k += 32)
        {
            _tile_loadd(7, B1 + k * 16, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_stream_loadd(5, A1 + k * 16, 64);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stream_loadd(4, A0 + k * 16 + 1024, 64);
            _tile_dpbf16ps(2, 5, 6);
            _tile_loadd(6, B0 + k * 16 + 1024, 64);
            _tile_dpbf16ps(3, 5, 7);
        }
        _tile_loadd(7, B1 + k * 16, 64);
        _tile_dpbf16ps(0, 4, 6);
        _tile_stored(0, C + 0, ldc * 4);
        _tile_stream_loadd(5, A1 + k * 16, 64);
        _tile_dpbf16ps(1, 4, 7);
        _tile_stored(1, C + 16, ldc * 4);
        _tile_dpbf16ps(2, 5, 6);
        _tile_stored(2, C + 16 * ldc + 0, ldc * 4);
        _tile_dpbf16ps(3, 5, 7);
        _tile_stored(3, C + 16 * ldc + 16, ldc * 4);
#endif
        for (int k = 0; k < K; k += 32)
        {
            _tile_stream_loadd(4, A0 + k * 16, 64);
            _tile_loadd(6, B0 + k * 16, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_loadd(7, B1 + k * 16, 64);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stream_loadd(5, A1 + k * 16, 64);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }

        _tile_stored(0, C + 0, ldc * 4);
        _tile_stored(1, C + 16, ldc * 4);
        _tile_stored(2, C + 16 * ldc + 0, ldc * 4);
        _tile_stored(3, C + 16 * ldc + 16, ldc * 4);


        return uint64_t(K * 2 * 32 * 32);
    }

    inline uint64_t PerfMacroBf16L3(int M, int N, int K, const uint16_t* A0, const uint16_t* B0, float* C, int ldc, int zero)
    {
        const uint16_t* A1 = A0 + 16 * K, * B1 = B0 + K * 16;
        uint64_t n = 0;
        for (int j = 0; j < N; j += 32)
        {
            for (int i = 0; i < M; i += 32)
                n += PerfMicroBf16L3(K, A0 + i * K, A1 + i * K, B0 + K * j, B1 + K * j, C + i * ldc + j, ldc, zero);
        }
        return n;
    }

    void TestPerfBf16L3(double time, int K)
    {
        std::cout << "Test L3 AMX BF16 performance: " << std::setprecision(3) << std::fixed << std::flush;

        TileConf conf;
        _tile_loadconfig(&conf);

        const int L1 = 48 * 1024, L2 = int(1 * 1024 * 1024), L3 = 1 * 1024 * 1024;
        //const int K = L1 / 2 / 32;
        const int M = L2 / 2 / K / 32 * 32;
        const int N = L3 / 2 / K / 32 * 32;

        Mat16b a(M, K), b(K, N);
        Mat32f c(M, N);
        Fill(a); Fill(b); Fill(c);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfMacroBf16L3(M, N, K, a.p, b.p, c.p, N, 1);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS. K = " << K << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfMicroBf16L3_2x2(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        //_mm_prefetch((const char*)A, _MM_HINT_T1);
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
        _tile_loadconfig(&conf);

        const int L1 = 48 * 1024 * 1, L2 = 1 * 1024 * 1024, L3 = 1 * 1024 * 1024;
        const int K = L1 / 2 / 32;
        const int M = L2 / 2 / K / 32 * 32;
        const int N = L3 / 2 / K / 32 * 32;

        Mat16b a(M, K), b(K, N);
        Mat32f c(M, N);
        Fill(a); Fill(b); Fill(c);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfMacroBf16L3_2x2(M, N, K, a.p, K, b.p, c.p, N, 1);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS. " << std::endl;
    }

    //-------------------------------------------------------------------------------------------------

    inline uint64_t PerfMicroBf16L3_2x2_v2(int K, const uint16_t* A0, const uint16_t* A1, const uint16_t* B0, const uint16_t* B1, float* C, int ldc, bool update)
    {
        if (update)
        {
            _tile_stream_loadd(0, C + 0, ldc * 4);
            _tile_stream_loadd(1, C + 16, ldc * 4);
            _tile_stream_loadd(2, C + 16 * ldc + 0, ldc * 4);
            _tile_stream_loadd(3, C + 16 * ldc + 16, ldc * 4);
        }
        else
        {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
        }
        for (size_t k = 0; k < K; k += 32)
        {
            _tile_stream_loadd(4, A0 + k * 16, 64);
            _tile_loadd(6, B0 + k * 16, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_loadd(7, B1 + k * 16, 64);
            _tile_dpbf16ps(1, 4, 7);
            _tile_stream_loadd(5, A1 + k * 16, 64);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }
        _tile_stored(0, C + 0, ldc * 4);
        _tile_stored(1, C + 16, ldc * 4);
        _tile_stored(2, C + 16 * ldc + 0, ldc * 4);
        _tile_stored(3, C + 16 * ldc + 16, ldc * 4);
        for (int i = 0; i < 32; ++i)
        {
            _mm_prefetch((const char*)(C + 00 + i * ldc), _MM_HINT_T2);
            _mm_prefetch((const char*)(C + 16 + i * ldc), _MM_HINT_T2);
        }

        return uint64_t(K * 2 * 32 * 32);
    }

    inline void PrefetchToL2(const uint16_t* ptr, size_t size)
    {
        for (int i = 0; i < size; i += 4, ptr += 128)
        {
            _mm_prefetch((const char*)(ptr + 0 * 32), _MM_HINT_T1);
            _mm_prefetch((const char*)(ptr + 1 * 32), _MM_HINT_T1);
            _mm_prefetch((const char*)(ptr + 2 * 32), _MM_HINT_T1);
            _mm_prefetch((const char*)(ptr + 3 * 32), _MM_HINT_T1);
        }
    }

    inline void PrefetchToL3(const uint16_t* ptr, size_t size)
    {
        for (int i = 0; i < size; i += 4, ptr += 128)
        {
            _mm_prefetch((const char*)(ptr + 0 * 32), _MM_HINT_T2);
            _mm_prefetch((const char*)(ptr + 1 * 32), _MM_HINT_T2);
            _mm_prefetch((const char*)(ptr + 2 * 32), _MM_HINT_T2);
            _mm_prefetch((const char*)(ptr + 3 * 32), _MM_HINT_T2);
        }
    }

    inline void PrefetchToMemory(const uint16_t* ptr, size_t size)
    {
        for (int i = 0; i < size; ++i)
            _mm_prefetch((const char*)(ptr + i * 32), _MM_HINT_NTA);
    }

    inline uint64_t PerfMacroBf16L3_2x2_v2(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C, int ldc, bool update)
    {
        uint64_t n = 0;
        for (int j = 0; j < N; j += 32)
        {
            const uint16_t* B0 = B + j * K, *B1 = B + (j + 16) * K;
            PrefetchToL2(B0 + K * 32, K);
            PrefetchToL2(B1 + K * 32, K);
            for (int i = 0; i < M; i += 32)
                n += PerfMicroBf16L3_2x2_v2(K, A + i * K, A + (i + 16) * K, B0, B1, C + i * ldc + j, ldc, update);
            PrefetchToL3(B0, K);
            PrefetchToL3(B1, K);
        }
        return n;
    }

    void TestPerfBf16L3_2x2_v2(double time, int M, int N)
    {
        std::cout << "Test L3 AMX BF16 2x2 performance: " << std::setprecision(3) << std::fixed;

        TileConf conf;
        _tile_loadconfig(&conf);

        const int L1 = 48 * 1024;
        const int K = L1 / 2 / 32;

        Mat16b a(M, K), b(K, N);
        Mat32f c(M, N);
        Fill(a); Fill(b); Fill(c);

        double t = 0;
        uint64_t n = 0;
        while (t < time)
        {
            double start = Time();
            n += PerfMacroBf16L3_2x2_v2(M, N, K, a.p, b.p, c.p, N, 0);
            t += Time() - start;
        }
        double gflops = double(n) / t / double(1024 * 1024 * 1024);
        std::cout << gflops << " GFLOPS. V2 " << M <<"-" << N << "-" << K << "; size (A + B + C) = " << (K * M + K * N /*+ M * N*/) * 2 / 1024 << " kB." << std::endl;
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
        TestPerfBf16L0(time, 32);
        TestPerfBf16L0(time, 10);
        TestPerfInt8L0(time);

        //for (int i = 1; i < 20000; i *= 2)
        //{
        //    TestLoadLongRows<2>(i * 4, time);
        //    TestLoadLongRows<2>(i * 5, time);
        //    TestLoadLongRows<2>(i * 6, time);
        //    TestLoadLongRows<2>(i * 7, time);
        //}

        //for (int i = 1; i < 20000; i *= 2)
        //{
        //    TestLoadCompact<2>(i * 4, time);
        //    TestLoadCompact<2>(i * 5, time);
        //    TestLoadCompact<2>(i * 6, time);
        //    TestLoadCompact<2>(i * 7, time);
        //}

        //for (int i = 4; i < 30000; i *= 2)
        //{
        //    TestPerfBf16L1<0 | 0>(i * 2, time);
        //    TestPerfBf16L1<0 | 0>(i * 3, time);
        //}

        //for (int i = 1; i < 30000; i *= 2)
        //{
        //    TestPerfBf16L1<1 | 8>(i * 2, time);
        //    TestPerfBf16L1<1 | 8>(i * 3, time);
        //}

        //for (int i = 1; i < 30000; i *= 2)
        //{
        //    TestPerfBf16L1<1 | 8 | 16>(i * 2, time);
        //    TestPerfBf16L1<1 | 8 | 16>(i * 3, time);
        //}

        //for (int i = 4; i < 30000; i *= 2)
        //{
        //    TestPerfBf16L1<2 | 8>(i * 2, time);
        //    TestPerfBf16L1<2 | 8>(i * 3, time);
        //}

        //TestPerfBf16L1_2x2(time);
        //TestPerfBf16L1_2x1(time);
        //TestPerfBf16L1_1x2(time);
        //TestPerfBf16L1_1x1(time);

        //TestPerfBf16L2(time);

        //for (int k = 32; k <= 4 * 1024; k *= 2)
        //{
        //    TestPerfBf16L3(time, k * 4);
        //    TestPerfBf16L3(time, k * 5);
        //    TestPerfBf16L3(time, k * 6);
        //    TestPerfBf16L3(time, k * 7);
        //}

        //TestPerfBf16L3_2x2(time);

        //for (int i = 1; i < 1000; i *= 2)
        //{
        //    TestPerfBf16L3_2x2_v2(time, 512, i * 32);
        //    if (i > 1)
        //        TestPerfBf16L3_2x2_v2(time, 512, i * 32);
        //}

        //for (int i = 1; i < 1000; i *= 2)
        //{
        //    TestPerfBf16L3_2x2_v2(time, 768, i * 32);
        //    if (i > 1)
        //        TestPerfBf16L3_2x2_v2(time, 768, i * 32);
        //}

        //for (int i = 1; i < 1000; i *= 2)
        //{
        //    TestPerfBf16L3_2x2_v2(time, 1024, i * 32);
        //    if (i > 1)
        //        TestPerfBf16L3_2x2_v2(time, 1024, i * 32);
        //}

        //for (int i = 1; i < 1000; i *= 2)
        //{
        //    for (int j = 1; j < 1000; j *= 2)
        //    {
        //        TestPerfBf16L3_2x2_v2(time, i * 32, j * 32);
        //        if (i > 1)
        //            TestPerfBf16L3_2x2_v2(time, i * 48, j * 32);
        //        if (j > 1)
        //            TestPerfBf16L3_2x2_v2(time, i * 32, j * 48);
        //        if (i > 1 && j > 1)
        //            TestPerfBf16L3_2x2_v2(time, i * 48, j * 48);
        //    }
        //}

        //TestPerfBf16L3_2x1(time);
        //TestPerfBf16L3_1x2(time);

        //TestPerfBf16L3_1xX<1>(time);
        //TestPerfBf16L3_1xX<2>(time);
        //TestPerfBf16L3_1xX<3>(time);
        //TestPerfBf16L3_1xX<4>(time);
        //TestPerfBf16L3_1xX<5>(time);
        //TestPerfBf16L3_1xX<6>(time);
    }

    void WarmUpCpu(double time)
    {
        std::cout << "Warm up CPU:" << std::endl;
        PrintCurrentFrequency();
        double t = 0;
        Amx::TileConf conf;
        _tile_loadconfig(&conf);
        while (t < time)
        {
            double start = Time();
            Amx::PerfBf16L0(1024*1024, 32);
            t += Time() - start;
        }
        PrintCurrentFrequency();
    }
}




