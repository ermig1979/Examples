#include "gemm.h"
#include "diff.h"
#include "time.h"
#include "amx.h"

namespace Amx
{
    static uint64_t PerfBf16L0(int count, const float * src, float * dst)
    {
        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

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
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "Test L0 AMX BF16 performance: ";

        Mat32f stub(8, 16 * 16);
        Fill(stub);
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
        std::cout << std::setprecision(3) << std::fixed << gflops << " GFLOPS; AMX BF16 time: " << optime << " nsec." << std::endl;
        if (stub.p[0])
            std::cout << " ";
    }

    //-------------------------------------------------------------------------------------------------

    static uint64_t PerfInt8L0(int count, const int32_t* src, int32_t* dst)
    {
        TileConf conf;
        conf.SetMax();
        _tile_loadconfig(&conf);

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

    void TestPerf(double time)
    {
        TestPerfBf16L0(time);

        TestPerfInt8L0(time);
    }
}


