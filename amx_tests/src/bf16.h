#pragma once

#include "defs.h"

union F32
{
    F32(float val) : f32{ val } {   }
    F32(uint32_t val) : u32{ val } {  }

    float f32;
    uint32_t u32;
};

inline float Round(float src)
{
    return F32((F32(src).u32 + 0x8000) & 0xFFFF0000).f32;
}

inline void Convert(float src, uint16_t& dst)
{
    dst = uint16_t((F32(src).u32 + 0x8000) >> 16);
}

inline void Convert(uint16_t src, float& dst)
{
    dst = F32(uint32_t(src) << 16).f32;
}

inline uint16_t To16b(float val)
{
    return uint16_t((F32(val).u32 + 0x8000) >> 16);
}

inline float To32f(uint16_t val)
{
    return F32(uint32_t(val) << 16).f32;
}
