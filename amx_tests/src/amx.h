#pragma once

#include "defs.h"

namespace Amx
{
    void InitAmx();

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
        }

        inline void SetMax()
        {
            rows[0] = 16;
            rows[1] = 16;
            rows[2] = 16;
            rows[3] = 16;
            rows[4] = 16;
            rows[5] = 16;
            rows[6] = 16;
            rows[7] = 16;
            colsb[0] = 64;
            colsb[1] = 64;
            colsb[2] = 64;
            colsb[3] = 64;
            colsb[4] = 64;
            colsb[5] = 64;
            colsb[6] = 64;
            colsb[7] = 64;
        }
    };

    //-------------------------------------------------------------------------------------------------

    void TestMaxBf16L0(double time = 1.0);

    void TestPerf(double time = 1.0);
}


