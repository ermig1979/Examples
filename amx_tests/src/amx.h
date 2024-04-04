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

    void TestMaxBf16L0(double time = 1.0);

    void TestPerf(double time = 1.0);
}


