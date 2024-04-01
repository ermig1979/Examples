#include "gemm.h"

namespace Avx512bw
{
    void Micro32f12x32(int K, const float* A, int lda, int step,
        const float* B, int ldb, float* C, int ldc)
    {
        const float* A0 = A, * A6 = A + 6 * lda;
        const size_t oa0 = lda * 0;
        const size_t oa1 = lda * 1;
        const size_t oa2 = lda * 2;
        const size_t oa3 = lda * 3;
        const size_t oa4 = lda * 4;
        const size_t oa5 = lda * 5;
        __m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, 
            c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1, b0, b1, a0;
        c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
        c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
        c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
        c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
        c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
        c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
        c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
        c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
        c80 = _mm512_setzero_ps(), c81 = _mm512_setzero_ps();
        c90 = _mm512_setzero_ps(), c91 = _mm512_setzero_ps();
        ca0 = _mm512_setzero_ps(), ca1 = _mm512_setzero_ps();
        cb0 = _mm512_setzero_ps(), cb1 = _mm512_setzero_ps();
        for (int k = 0; k < K; k++)
        {
            b0 = _mm512_loadu_ps(B + 0);
            b1 = _mm512_loadu_ps(B + 16);
            a0 = _mm512_set1_ps(A0[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01);
            a0 = _mm512_set1_ps(A0[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11);
            a0 = _mm512_set1_ps(A0[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21);
            a0 = _mm512_set1_ps(A0[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31);
            a0 = _mm512_set1_ps(A0[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41);
            a0 = _mm512_set1_ps(A0[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51);
            a0 = _mm512_set1_ps(A6[oa0]), c60 = _mm512_fmadd_ps(a0, b0, c60), c61 = _mm512_fmadd_ps(a0, b1, c61);
            a0 = _mm512_set1_ps(A6[oa1]), c70 = _mm512_fmadd_ps(a0, b0, c70), c71 = _mm512_fmadd_ps(a0, b1, c71);
            a0 = _mm512_set1_ps(A6[oa2]), c80 = _mm512_fmadd_ps(a0, b0, c80), c81 = _mm512_fmadd_ps(a0, b1, c81);
            a0 = _mm512_set1_ps(A6[oa3]), c90 = _mm512_fmadd_ps(a0, b0, c90), c91 = _mm512_fmadd_ps(a0, b1, c91);
            a0 = _mm512_set1_ps(A6[oa4]), ca0 = _mm512_fmadd_ps(a0, b0, ca0), ca1 = _mm512_fmadd_ps(a0, b1, ca1);
            a0 = _mm512_set1_ps(A6[oa5]), cb0 = _mm512_fmadd_ps(a0, b0, cb0), cb1 = _mm512_fmadd_ps(a0, b1, cb1);
            B += ldb; A0 += step, A6 += step;
        }
        _mm512_storeu_ps(C + 0, _mm512_add_ps(c00, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(c01, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(c10, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(c11, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(c20, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(c21, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(c30, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(c31, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(c40, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(c41, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(c50, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(c51, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(c60, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(c61, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(c70, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(c71, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(c80, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(c81, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(c90, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(c91, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(ca0, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(ca1, _mm512_loadu_ps(C + 16)));
        C += ldc;
        _mm512_storeu_ps(C + 0, _mm512_add_ps(cb0, _mm512_loadu_ps(C + 0)));
        _mm512_storeu_ps(C + 16, _mm512_add_ps(cb1, _mm512_loadu_ps(C + 16)));
    }

    void InitC(int M, int N, float* C, int ldc)
    {
        for (int i = 0; i < M; ++i, C += ldc)
            for (int j = 0; j < N; j += 16)
                _mm512_storeu_ps(C + j, _mm512_setzero_ps());
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

    void Reorder32fA12(const float* A, int lda, int M, int K, float* bufA)
    {
        for (int i = 0; i < M; i += 12)
        {
            for (int k = 0; k < K; k += 4)
            {
                const float* pA = A + k;
                Transpose4x4(pA + 0 * lda, lda, bufA + 0, 12);
                Transpose4x4(pA + 4 * lda, lda, bufA + 4, 12);
                Transpose4x4(pA + 8 * lda, lda, bufA + 8, 12);
                bufA += 48;
            }
            A += 12 * lda;
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
        int ldb, float* bufB, bool reorderB, float* C, int ldc)
    {
        for (int j = 0; j < N; j += 32)
        {
            if (reorderB)
                Reorder32fB32(K, B + j, ldb, bufB + K * j);
            for (int i = 0; i < M; i += 12)
                Micro32f12x32(K, A + i * K, 1, 12, bufB + K * j, 32, C + i * ldc + j, ldc);
        }
    }

    void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C)
    {
        const int L1 = 48 * 1024, L2 = 2 * 1024 * 1024, L3 = 2 * 1024 * 1024;
        int mK = std::min(L1 / 4 / 32, K) / 4 * 4;
        int mM = std::min(L2 / 4 / mK, M) / 12 * 12;
        int mN = std::min(L3 / 4 / mK, N) / 32 * 32;
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
                    if (k == 0)
                        InitC(dM, dN, C + i * N + j, N);
                    Reorder32fA12(A + i * K + k, K, dM, dK, bufA.p);
                    Macro32f(dM, dN, dK, bufA.p, B + k * N + j, N, bufB.p, i == 0, C + i * N + j, N);
                }
            }
        }
    }
}

