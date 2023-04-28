#pragma once

#include "defs.h"

namespace cs
{
    namespace Fp16
    {
        union Bits
        {
            float f;
            int32_t si;
            uint32_t ui;
        };

        const int SHIFT = 13;
        const int SHIFT_SIGN = 16;

        const int32_t INF_N = 0x7F800000; // flt32 infinity
        const int32_t MAX_N = 0x477FE000; // max flt16 normal as a flt32
        const int32_t MIN_N = 0x38800000; // min flt16 normal as a flt32
        const int32_t SIGN_N = 0x80000000; // flt32 sign bit

        const int32_t INF_C = INF_N >> SHIFT;
        const int32_t NAN_N = (INF_C + 1) << SHIFT; // minimum flt16 nan as a flt32
        const int32_t MAX_C = MAX_N >> SHIFT;
        const int32_t MIN_C = MIN_N >> SHIFT;
        const int32_t SIGN_C = SIGN_N >> SHIFT_SIGN; // flt16 sign bit

        const int32_t MUL_N = 0x52000000; // (1 << 23) / MIN_N
        const int32_t MUL_C = 0x33800000; // MIN_N / (1 << (23 - shift))

        const int32_t SUB_C = 0x003FF; // max flt32 subnormal down shifted
        const int32_t NOR_C = 0x00400; // min flt32 normal down shifted

        const int32_t MAX_D = INF_C - MAX_C - 1;
        const int32_t MIN_D = MIN_C - SUB_C - 1;
    }

    inline uint16_t Float32ToFloat16(float value)
    {
        Fp16::Bits v, s;
        v.f = value;
        uint32_t sign = v.si & Fp16::SIGN_N;
        v.si ^= sign;
        sign >>= Fp16::SHIFT_SIGN; // logical shift
        s.si = Fp16::MUL_N;
        s.si = int32_t(s.f * v.f); // correct subnormals
        v.si ^= (s.si ^ v.si) & -(Fp16::MIN_N > v.si);
        v.si ^= (Fp16::INF_N ^ v.si) & -((Fp16::INF_N > v.si) & (v.si > Fp16::MAX_N));
        v.si ^= (Fp16::NAN_N ^ v.si) & -((Fp16::NAN_N > v.si) & (v.si > Fp16::INF_N));
        v.ui >>= Fp16::SHIFT; // logical shift
        v.si ^= ((v.si - Fp16::MAX_D) ^ v.si) & -(v.si > Fp16::MAX_C);
        v.si ^= ((v.si - Fp16::MIN_D) ^ v.si) & -(v.si > Fp16::SUB_C);
        return v.ui | sign;
    }

    inline float Float16ToFloat32(uint16_t value)
    {
        Fp16::Bits v;
        v.ui = value;
        int32_t sign = v.si & Fp16::SIGN_C;
        v.si ^= sign;
        sign <<= Fp16::SHIFT_SIGN;
        v.si ^= ((v.si + Fp16::MIN_D) ^ v.si) & -(v.si > Fp16::SUB_C);
        v.si ^= ((v.si + Fp16::MAX_D) ^ v.si) & -(v.si > Fp16::MAX_C);
        Fp16::Bits s;
        s.si = Fp16::MUL_C;
        s.f *= v.si;
        int32_t mask = -(Fp16::NOR_C > v.si);
        v.si <<= Fp16::SHIFT;
        v.si ^= (s.si ^ v.si) & mask;
        v.si |= sign;
        return v.f;
    }
}


