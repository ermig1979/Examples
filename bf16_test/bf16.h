#pragma once

#include "defs.h"


union F32 
{
    F32(float val) : f32{ val }  {   }
    F32(uint32_t val) : u32{ val } {  }

    float f32;
    uint32_t u32;
};

inline float bf16_original(float val)
{
    return val;
}

inline float bf16_truncate(float val)
{
    return F32(F32(val).u32 & 0xFFFF0000).f32;
}

inline float bf16_nearest(float val)
{
    return F32((F32(val).u32 + 0x8000) & 0xFFFF0000).f32;
}

inline float bf16_nearest_even(float val)
{
    return F32((F32(val).u32 + ((F32(val).u32 & 0x00010000) >> 1)) & 0xFFFF0000).f32; 
}

inline float bf16_magic_number(float val)
{
    return F32(F32(val * (1.000000f + 0.00390625f /*0.001957f*/)).u32 & 0xFFFF0000).f32;
}