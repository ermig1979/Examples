#include "test.h"
#include "mat.h"
#include "amx.h"

namespace Amx
{

    void Micro16b16x16(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        TileConf conf;
        _tile_loadconfig(&conf);

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
            _tile_stream_loadd(4, A + k, lda * 2);
            _tile_loadd(6, B + k * 16 + 0, 64);
            _tile_dpbf16ps(0, 4, 6);
        }
        _tile_stored(0, C + 0, ldc * 4);
    }

    void Micro16b32x16(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        TileConf conf;
        _tile_loadconfig(&conf);

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
    }

    void Micro16b16x32(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        TileConf conf;
        _tile_loadconfig(&conf);

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
    }


    void Micro16b32x32(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        TileConf conf;
        _tile_loadconfig(&conf);

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
    }

    inline void ConvertA(const float* src, uint16_t* dst)
    {
        __m512 s0 = _mm512_loadu_ps(src + 0 * 16);
        __m512 s1 = _mm512_loadu_ps(src + 1 * 16);
        _mm512_storeu_si512(dst, (__m512i)_mm512_cvtne2ps_pbh(s1, s0));
    }

    static void ConvertA(const float* A, int lda, int M, int K, uint16_t* bufA, int bufS)
    {
        for (int i = 0; i < M; i += 1)
        {
            for (int k = 0; k < K; k += 32)
                ConvertA(A + k, bufA + k);
            A += lda;
            bufA += bufS;
        }
    }

    inline void ReorderB(const float* src, int stride, uint16_t* dst)
    {
        static const __m512i PERM_IDX = _mm512_set_epi16(
            0x1f, 0x0f, 0x1e, 0x0e, 0x1d, 0x0d, 0x1c, 0x0c, 0x1b, 0x0b, 0x1a, 0x0a, 0x19, 0x09, 0x18, 0x08, 
            0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14, 0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01, 0x10, 0x00);
        __m512 s0 = _mm512_loadu_ps(src + 0 * stride);
        __m512 s1 = _mm512_loadu_ps(src + 1 * stride);
        __m512i d = (__m512i)_mm512_cvtne2ps_pbh(s1, s0);
        _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(PERM_IDX, d));
    }

    void ReorderB32(int K, const float* B, int ldb, uint16_t* bufB)
    {
        for (int k = 0; k < K; k += 2, B += 2 * ldb, bufB += 64)
        {
            ReorderB(B + 0, ldb, bufB + 0);
            ReorderB(B + 16, ldb, bufB + 32);
        }
    }

    void Macro32f(int M, int N, int K, const uint16_t* A, int lda, const float* B,
        int ldb, uint16_t* bufB, bool reorderB, float* C, int ldc, int zero)
    {
        for (int j = 0; j < N; j += 32)
        {
            if (reorderB)
                ReorderB32(K, B + j, ldb, bufB + K * j);
            for (int i = 0; i < M; i += 32)
                Micro16b32x32(K, A + i * lda, lda, bufB + K * j, C + i * ldc + j, ldc, zero);
        }
    }

    void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C)
    {
        const int L1 = 48 * 1024, L2 = 1 * 1024 * 1024, L3 = 2 * 1024 * 1024;
        int mK = std::min(L1 / 2 / 32, K) / 32 * 32;
        int mM = std::min(L2 / 2 / mK, M) / 32 * 32;
        int mN = std::min(L3 / 2 / mK, N) / 32 * 32;
        Mat16b bufB(mN, mK);
        Mat16b bufA(mK, mM);
        for (int j = 0; j < N; j += mN)
        {
            int dN = std::min(N, j + mN) - j;
            for (int k = 0; k < K; k += mK)
            {
                int dK = std::min(K, k + mK) - k;
                for (int i = 0; i < M; i += mM)
                {
                    int dM = std::min(M, i + mM) - i;
                    ConvertA(A + i * K + k, K, dM, dK, bufA.p, mK);
                    Macro32f(dM, dN, dK, bufA.p, mK, B + k * N + j, N, bufB.p, i == 0, C + i * N + j, N, k == 0);
                }
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    void ReorderB(int macroN, int macroK, int microN, const float* src, int stride, uint16_t* dst)
    {
        for (int j = 0; j < macroN; j += microN)
        {
            for (int k = 0; k < macroK; k += 2)
            {
                for(int m = 0; m < microN; m += 16)
                    ReorderB(src + k * stride + m, stride, dst + k * microN + macroK * j + m * 2);
            }
            src += microN;
        }
    }

    void ReorderB(int N, int K, int microN, const float* src, uint16_t* dst)
    {
        const int L1 = 48 * 1024, L3 = 2 * 1024 * 1024;
        int macroK = AlignLo(Min(L1 / 2 / microN, K), microN);
        int macroN = AlignLo(Min(L3 / 2 / macroK, N), microN);
        for (int j = 0; j < N; j += macroN)
        {
            int dN = Min(N, j + macroN) - j;
            for (int k = 0; k < K; k += macroK)
            {
                int dK = Min(K, k + macroK) - k;
                ReorderB(dN, dK, microN, src + k * N + j, N, dst);
                dst += dN * dK;
            }
        }
    }

    void Macro16b(int M, int N, int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        for (int j = 0; j < N; j += 32)
        {
            for (int i = 0; i < M; i += 32)
                Micro16b32x32(K, A + i * lda, lda, B + K * j, C + i * ldc + j, ldc, zero);
        }
    }

    void Gemm32f16b(int M, int N, int K, const float* A, const uint16_t* B, float* C)
    {
        const int L1 = 48 * 1024, L2 = 1 * 1024 * 1024, L3 = 2 * 1024 * 1024;
        int mK = std::min(L1 / 2 / 32, K) / 32 * 32;
        int mM = std::min(L2 / 2 / mK, M) / 32 * 32;
        int mN = std::min(L3 / 2 / mK, N) / 32 * 32;
        Mat16b bufA(mK, mM);
        for (int j = 0; j < N; j += mN)
        {
            int dN = std::min(N, j + mN) - j;
            for (int k = 0; k < K; k += mK)
            {
                int dK = std::min(K, k + mK) - k;
                for (int i = 0; i < M; i += mM)
                {
                    int dM = std::min(M, i + mM) - i;
                    ConvertA(A + i * K + k, K, dM, dK, bufA.p , mK);
                    Macro16b(dM, dN, dK, bufA.p, mK, B, C + i * N + j, N, k == 0);
                }
                B += dN * dK;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    void Gemm32f16bV2(int M, int N, int K, const float* A, const uint16_t* B, float* C)
    {
        const int L1 = 48 * 1024, L2 = 1 * 1024 * 1024, L3 = 2 * 1024 * 1024;
        int mK = std::min(L1 / 2 / 32, K) / 32 * 32;
        int mM = std::min(L2 / 2 / mK, M) / 32 * 32;
        int mN = std::min(L3 / 2 / mK, N) / 32 * 32;
        Mat16b bufA(K, M);
        for (int j = 0; j < N; j += mN)
        {
            int dN = std::min(N, j + mN) - j;
            for (int k = 0; k < K; k += mK)
            {
                int dK = std::min(K, k + mK) - k;
                for (int i = 0; i < M; i += mM)
                {
                    int dM = std::min(M, i + mM) - i;
                    if (j == 0 && k == 0)
                    {
                        for (int k = 0; k < K; k += mK)
                        {
                            int dK = std::min(K, k + mK) - k;
                            ConvertA(A + i * K + k, K, dM, dK, bufA.p + k * M + i * mK, mK);
                        }
                    }
                    Macro16b(dM, dN, dK, bufA.p + k * M + i * mK, mK, B, C + i * N + j, N, k == 0);
                }
                B += dN * dK;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    void Macro16bV3(int M, int N, int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        for (int j = 0; j < N; j += 16)
        {
            for (int i = 0; i < M; i += 32)
                Micro16b32x16(K, A + i * lda, lda, B + K * j, C + i * ldc + j, ldc, zero);
        }
    }

    void Gemm32f16bV3(int M, int N, int K, const float* A, const uint16_t* B, float* C)
    {
        const int L1 = 48 * 1024, L2 = 1 * 1024 * 1024, L3 = 2 * 1024 * 1024;
        int mK = AlignLo(Min(L1 / 2 / 16, K), 16);
        int mM = AlignLo(Min(L2 / 2 / mK, M), 32);
        int mN = AlignLo(Min(L3 / 2 / mK, N), 16);
        Mat16b bufA(K, M);
        for (int j = 0; j < N; j += mN)
        {
            int dN = std::min(N, j + mN) - j;
            for (int k = 0; k < K; k += mK)
            {
                int dK = std::min(K, k + mK) - k;
                for (int i = 0; i < M; i += mM)
                {
                    int dM = std::min(M, i + mM) - i;
                    if (j == 0 && k == 0)
                    {
                        for (int k = 0; k < K; k += mK)
                        {
                            int dK = std::min(K, k + mK) - k;
                            ConvertA(A + i * K + k, K, dM, dK, bufA.p + k * M + i * mK, mK);
                        }
                    }
                    Macro16bV3(dM, dN, dK, bufA.p + k * M + i * mK, mK, B, C + i * N + j, N, k == 0);
                }
                B += dN * dK;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    void ConvertA(int M, int K, const float* src, uint16_t* dst)
    {
        for (int i = 0, n = M * K; i < n; i += 32)
            ConvertA(src + i, dst + i);
    }

    void ReorderA(int M, int K, const float* src, uint16_t* dst)
    {
        for (int i = 0, n = M * K; i < n; i += 32)
            ConvertA(src + i, dst + i);
    }

    static void ReorderA(const uint16_t* A, int lda, int M, int K, uint16_t* bufA, int bufS)
    {
        for (int i = 0; i < M; i += 1)
        {
            memcpy(bufA, A, K * 2);
            A += lda;
            bufA += bufS;
        }
    }

    void Gemm16b(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C)
    {
        const int L1 = 48 * 1024, L2 = 1 * 1024 * 1024, L3 = 2 * 1024 * 1024;
        int mK = std::min(L1 / 2 / 32, K) / 32 * 32;
        int mM = std::min(L2 / 2 / mK, M) / 32 * 32;
        int mN = std::min(L3 / 2 / mK, N) / 32 * 32;
        Mat16b bufA(K, M);
        for (int j = 0; j < N; j += mN)
        {
            int dN = std::min(N, j + mN) - j;
            for (int k = 0; k < K; k += mK)
            {
                int dK = std::min(K, k + mK) - k;
                for (int i = 0; i < M; i += mM)
                {
                    int dM = std::min(M, i + mM) - i;
                    uint16_t* pA = mK < K ? bufA.p : (uint16_t*)A;
                    if (j == 0 && k == 0 && mK < K)
                    {
                        for (int k = 0; k < K; k += mK)
                        {
                            int dK = std::min(K, k + mK) - k;
                            ReorderA(A + i * K + k, K, dM, dK, bufA.p + k * M + i * mK, mK);
                        }
                    }
                    Macro16b(dM, dN, dK, pA + k * M + i * mK, mK, B, C + i * N + j, N, k == 0);
                }
                B += dN * dK;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    void StubMicro16b(int M, int N, int K, const float* A, const float* B, float* C)
    {
        const int L1 = 48 * 1024 / 2;
        int mK = std::min(L1 / 2 / 32 / 2, K) / 32 * 32;
        Mat16b a16b(32, K), b16b(K, 32);
        Fill(a16b), Fill(b16b);
        for (int i = 0; i < M; i += 32)
        {
            for (int j = 0; j < N; j += 32)
            {
                for (int k = 0; k < K; k += mK)
                {
                    int dK = std::min(mK, K - k);
                    Micro16b32x32(dK, a16b.p, mK, b16b.p, C, 32, i == 0 && j == 0 && k == 0);
                }
            }
        }
    }

    void StubMacro16b(int M, int N, int K, const float* A, const float* B, float* C)
    {
        const int L1 = 48 * 1024, L2 = 1 * 1024 * 1024, L3 = 2 * 1024 * 1024;
        int mK = std::min(L1 / 2 / 32, K) / 32 * 32;
        int mM = std::min(L2 / 2 / mK, M) / 32 * 32;
        int mN = std::min(L3 / 2 / mK, N) / 32 * 32;
        Mat16b a16b(mK, mM), b16b(mN, mK);
        Fill(a16b), Fill(b16b);

        for (int j = 0; j < N; j += mN)
        {
            int dN = std::min(N, j + mN) - j;
            for (int k = 0; k < K; k += mK)
            {
                int dK = std::min(K, k + mK) - k;
                for (int i = 0; i < M; i += mM)
                {
                    int dM = std::min(M, i + mM) - i;
                    Macro32f(dM, dN, dK, a16b.p, mK, B + k * N + j, N, b16b.p, 0, C, N, k == 0);
                }
            }
        }
    }
}

