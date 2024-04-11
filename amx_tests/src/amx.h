#pragma once

#include "defs.h"

namespace Amx
{
    inline void InitAmx()
    {
#if defined(__linux__)
        const int ARCH_GET_XCOMP_PERM = 0x1022;
        const int ARCH_REQ_XCOMP_PERM = 0x1023;
        const int XFEATURE_XTILECFG = 17;
        const int XFEATURE_XTILEDATA = 18;
        if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0)
            std::cout << "Can't initialize AMX!" << std::endl;
#endif
    }

    //-------------------------------------------------------------------------------------------------

    const size_t TILE_SIZE = 1024;
    const size_t BF16_OPS = 2 * 32 * 16 * 16;
    const size_t INT8_OPS = 2 * 64 * 16 * 16;

    //-------------------------------------------------------------------------------------------------

    struct TileConf
    {
        uint8_t paletteId;
        uint8_t startRow;
        uint8_t reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];

        inline TileConf(uint8_t paletteId = 1, uint8_t startRow = 0)
        {
            uint64_t* dst = (uint64_t*)this;
            for (size_t i = 0; i < 8; ++i)
                dst[i] = 0;
            this->paletteId = paletteId;
            this->startRow = startRow;
            for (size_t i = 0; i < 8; ++i)
            {
                rows[i] = 16;
                colsb[i] = 64;
            }
        }
    };

    //-------------------------------------------------------------------------------------------------

    inline void ConvertA(const float* src, uint16_t* dst)
    {
        __m512 s0 = _mm512_loadu_ps(src + 0 * 16);
        __m512 s1 = _mm512_loadu_ps(src + 1 * 16);
        _mm512_storeu_si512(dst, (__m512i)_mm512_cvtne2ps_pbh(s1, s0));
    }

    inline void ConvertB(const float* src, int stride, uint16_t* dst)
    {
        static const __m512i PERM_IDX = _mm512_set_epi16(
            0x1f, 0x0f, 0x1e, 0x0e, 0x1d, 0x0d, 0x1c, 0x0c, 0x1b, 0x0b, 0x1a, 0x0a, 0x19, 0x09, 0x18, 0x08,
            0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14, 0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01, 0x10, 0x00);
        __m512 s0 = _mm512_loadu_ps(src + 0 * stride);
        __m512 s1 = _mm512_loadu_ps(src + 1 * stride);
        __m512i d = (__m512i)_mm512_cvtne2ps_pbh(s1, s0);
        _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(PERM_IDX, d));
    }

    //-------------------------------------------------------------------------------------------------

    void TestMaxBf16L0(double time = 1.0);

    void TestPerf(double time = 1.0);

    void WarmUpCpu(double time = 1.0);
}


