#include "gemm.h"
#include "amx.h"

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18
#endif

namespace Amx
{
    void InitAmx()
    {
#if defined(__linux__)
        if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0)
            std::cout << "Can't initialize AMX!" << std::endl;
#endif
    }

    void Micro16b32x32(int K, const uint16_t* A, int lda, const uint16_t* B, float* C, int ldc, int zero)
    {
        TileConf conf;
        conf.SetMax();
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
        static const __m512i P0 = _mm512_set_epi16(0x2f, 0x0f, 0x2e, 0x0e, 0x2d, 0x0d, 0x2c, 0x0c, 0x2b, 0x0b, 0x2a, 0x0a, 0x29, 0x09, 0x28, 0x08, 0x27, 0x07, 0x26, 0x06, 0x25, 0x05, 0x24, 0x04, 0x23, 0x03, 0x22, 0x02, 0x21, 0x01, 0x20, 0x00);
        static const __m512i P1 = _mm512_set_epi16(0x3f, 0x1f, 0x3e, 0x1e, 0x3d, 0x1d, 0x3c, 0x1c, 0x3b, 0x1b, 0x3a, 0x1a, 0x39, 0x19, 0x38, 0x18, 0x37, 0x17, 0x36, 0x16, 0x35, 0x15, 0x34, 0x14, 0x33, 0x13, 0x32, 0x12, 0x31, 0x11, 0x30, 0x10);
        __m512 s00 = _mm512_loadu_ps(src + 0 * stride);
        __m512 s01 = _mm512_loadu_ps(src + 0 * stride + 16);
        __m512i d0 = (__m512i)_mm512_cvtne2ps_pbh(s01, s00);
        __m512 s10 = _mm512_loadu_ps(src + 1 * stride);
        __m512 s11 = _mm512_loadu_ps(src + 1 * stride + 16);
        __m512i d1 = (__m512i)_mm512_cvtne2ps_pbh(s11, s10);
        _mm512_storeu_si512(dst + 0, _mm512_permutex2var_epi16(d0, P0, d1));
        _mm512_storeu_si512(dst + 32, _mm512_permutex2var_epi16(d0, P1, d1));
    }

    void ReorderB(int K, const float* B, int ldb, uint16_t* bufB)
    {
        for (int k = 0; k < K; k += 2, B += 2 * ldb, bufB += 64)
            ReorderB(B, ldb, bufB);
    }

    void Macro32f(int M, int N, int K, const uint16_t* A, int lda, const float* B,
        int ldb, uint16_t* bufB, bool reorderB, float* C, int ldc, int zero)
    {
        for (int j = 0; j < N; j += 32)
        {
            if (reorderB)
                ReorderB(K, B + j, ldb, bufB + K * j);
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

    void ReorderB(int mN, int mK, const float* src, int stride, uint16_t* dst)
    {
        for (int j = 0; j < mN; j += 32)
            for (int k = 0; k < mK; k += 2)
                ReorderB(src + k * stride + j, stride, dst + k * 32 + mK * j);
    }

    void ReorderB(int N, int K, const float* src, uint16_t* dst)
    {
        const int L1 = 48 * 1024, L3 = 2 * 1024 * 1024;
        int mK = std::min(L1 / 2 / 32, K) / 32 * 32;
        int mN = std::min(L3 / 2 / mK, N) / 32 * 32;
        for (int j = 0; j < N; j += mN)
        {
            int dN = std::min(N, j + mN) - j;
            for (int k = 0; k < K; k += mK)
            {
                int dK = std::min(K, k + mK) - k;
                ReorderB(dN, dK, src + k * N + j, N, dst);
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
                    ConvertA(A + i * K + k, K, dM, dK, bufA.p, mK);
                    Macro16b(dM, dN, dK, bufA.p, mK, B, C + i * N + j, N, k == 0);
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

