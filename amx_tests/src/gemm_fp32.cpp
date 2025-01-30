#include "test.h"

namespace Base
{
    void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C)
    {
        for (int i = 0; i < M; ++i)
        {
            float* c = C + i * N;
            for (int j = 0; j < N; ++j)
                c[j] = 0;
            for (int k = 0; k < K; ++k)
            {
                const float* b = B + k * N;
                float a = A[i * K + k];
                for (int j = 0; j < N; ++j)
                    c[j] += a * b[j];
            }
        }
    }

    void Gemm16b(int M, int N, int K, const float* A, const float* B, float* C)
    {
        for (int i = 0; i < M; ++i)
        {
            float* c = C + i * N;
            for (int j = 0; j < N; ++j)
                c[j] = 0;
            for (int k = 0; k < K; ++k)
            {
                const float* b = B + k * N;
                float a = Round(A[i * K + k]);
                for (int j = 0; j < N; ++j)
                    c[j] += a * Round(b[j]);
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------

namespace Avx512bw
{
    void Micro32f8x32(int K, const float* A, int lda, int step,
        const float* B, int ldb, float* C, int ldc, int zero)
    {
        const float* A0 = A, * A4 = A + 4 * lda;
        const int oa0 = lda * 0;
        const int oa1 = lda * 1;
        const int oa2 = lda * 2;
        const int oa3 = lda * 3;
        __m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, 
            c60, c61, c70, c71, b0, b1, a0;
        if (zero)
        {
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
        }
        else
        {
            c00 = _mm512_loadu_ps(C + 0x0 * ldc + 0), c01 = _mm512_loadu_ps(C + 0x0 * ldc + 16);
            c10 = _mm512_loadu_ps(C + 0x1 * ldc + 0), c11 = _mm512_loadu_ps(C + 0x1 * ldc + 16);
            c20 = _mm512_loadu_ps(C + 0x2 * ldc + 0), c21 = _mm512_loadu_ps(C + 0x2 * ldc + 16);
            c30 = _mm512_loadu_ps(C + 0x3 * ldc + 0), c31 = _mm512_loadu_ps(C + 0x3 * ldc + 16);
            c40 = _mm512_loadu_ps(C + 0x4 * ldc + 0), c41 = _mm512_loadu_ps(C + 0x4 * ldc + 16);
            c50 = _mm512_loadu_ps(C + 0x5 * ldc + 0), c51 = _mm512_loadu_ps(C + 0x5 * ldc + 16);
            c60 = _mm512_loadu_ps(C + 0x6 * ldc + 0), c61 = _mm512_loadu_ps(C + 0x6 * ldc + 16);
            c70 = _mm512_loadu_ps(C + 0x7 * ldc + 0), c71 = _mm512_loadu_ps(C + 0x7 * ldc + 16);
        }
        for (int k = 0; k < K; k++)
        {
            b0 = _mm512_loadu_ps(B + 0);
            b1 = _mm512_loadu_ps(B + 16);
            a0 = _mm512_set1_ps(A0[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01);
            a0 = _mm512_set1_ps(A0[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11);
            a0 = _mm512_set1_ps(A0[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21);
            a0 = _mm512_set1_ps(A0[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31);
            a0 = _mm512_set1_ps(A4[oa0]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41);
            a0 = _mm512_set1_ps(A4[oa1]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51);
            a0 = _mm512_set1_ps(A4[oa2]), c60 = _mm512_fmadd_ps(a0, b0, c60), c61 = _mm512_fmadd_ps(a0, b1, c61);
            a0 = _mm512_set1_ps(A4[oa3]), c70 = _mm512_fmadd_ps(a0, b0, c70), c71 = _mm512_fmadd_ps(a0, b1, c71);
            B += ldb; A0 += step, A4 += step;
        }
        _mm512_storeu_ps(C + 0x0 * ldc + 0, c00), _mm512_storeu_ps(C + 0x0 * ldc + 16, c01);
        _mm512_storeu_ps(C + 0x1 * ldc + 0, c10), _mm512_storeu_ps(C + 0x1 * ldc + 16, c11);
        _mm512_storeu_ps(C + 0x2 * ldc + 0, c20), _mm512_storeu_ps(C + 0x2 * ldc + 16, c21);
        _mm512_storeu_ps(C + 0x3 * ldc + 0, c30), _mm512_storeu_ps(C + 0x3 * ldc + 16, c31);
        _mm512_storeu_ps(C + 0x4 * ldc + 0, c40), _mm512_storeu_ps(C + 0x4 * ldc + 16, c41);
        _mm512_storeu_ps(C + 0x5 * ldc + 0, c50), _mm512_storeu_ps(C + 0x5 * ldc + 16, c51);
        _mm512_storeu_ps(C + 0x6 * ldc + 0, c60), _mm512_storeu_ps(C + 0x6 * ldc + 16, c61);
        _mm512_storeu_ps(C + 0x7 * ldc + 0, c70), _mm512_storeu_ps(C + 0x7 * ldc + 16, c71);
    }

    inline void Transpose4x4(const float* src, size_t srcStride, float* dst, size_t dstStride)
    {
        __m128 s0 = _mm_loadu_ps(src + 0 * srcStride);
        __m128 s1 = _mm_loadu_ps(src + 1 * srcStride);
        __m128 s2 = _mm_loadu_ps(src + 2 * srcStride);
        __m128 s3 = _mm_loadu_ps(src + 3 * srcStride);
        __m128 s00 = _mm_unpacklo_ps(s0, s2);
        __m128 s01 = _mm_unpacklo_ps(s1, s3);
        __m128 s10 = _mm_unpackhi_ps(s0, s2);
        __m128 s11 = _mm_unpackhi_ps(s1, s3);
        _mm_storeu_ps(dst + 0 * dstStride, _mm_unpacklo_ps(s00, s01));
        _mm_storeu_ps(dst + 1 * dstStride, _mm_unpackhi_ps(s00, s01));
        _mm_storeu_ps(dst + 2 * dstStride, _mm_unpacklo_ps(s10, s11));
        _mm_storeu_ps(dst + 3 * dstStride, _mm_unpackhi_ps(s10, s11));
    }

    void Reorder32fA8(const float* A, int lda, int M, int K, float* bufA)
    {
        for (int i = 0; i < M; i += 8)
        {
            for (int k = 0; k < K; k += 4)
            {
                const float* pA = A + k;
                Transpose4x4(pA + 0 * lda, lda, bufA + 0, 8);
                Transpose4x4(pA + 4 * lda, lda, bufA + 4, 8);
                bufA += 32;
            }
            A += 8 * lda;
        }
    }

    void Reorder32fB32(int K, const float* B, int ldb, float* bufB)
    {
        for (int k = 0; k < K; ++k, B += ldb, bufB += 32)
        {
            _mm512_storeu_ps(bufB + 0, _mm512_loadu_ps(B + 0));
            _mm512_storeu_ps(bufB + 16, _mm512_loadu_ps(B + 16));
        }
    }

    void Macro32f(int M, int N, int K, const float* A, const float* B, 
        int ldb, float* bufB, bool reorderB, float* C, int ldc, int zero)
    {
        for (int j = 0; j < N; j += 32)
        {
            if (reorderB)
                Reorder32fB32(K, B + j, ldb, bufB + K * j);
            for (int i = 0; i < M; i += 8)
                Micro32f8x32(K, A + i * K, 1, 8, bufB + K * j, 32, C + i * ldc + j, ldc, zero);
        }
    }

    void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C)
    {
        const int L1 = Cache(1), L2 = Cache(2), L3 = Cache(3);
        int mK = AlignLo(Min(L1 / 4 / 32, K), 4);
        int mM = AlignLo(Min(L2 / 4 / mK, M), 8);
        int mN = AlignLo(Min(L3 / 4 / mK, N), 32);
        Mat32f bufB(mN, mK);
        Mat32f bufA(mK, mM);
        for (int j = 0; j < N; j += mN)
        {
            int dN = std::min(N, j + mN) - j;
            for (int k = 0; k < K; k += mK)
            {
                int dK = std::min(K, k + mK) - k;
                for (int i = 0; i < M; i += mM)
                {
                    int dM = std::min(M, i + mM) - i;
                    Reorder32fA8(A + i * K + k, K, dM, dK, bufA.p);
                    Macro32f(dM, dN, dK, bufA.p, B + k * N + j, N, bufB.p, i == 0, C + i * N + j, N, k == 0);
                }
            }
        }
    }
}

