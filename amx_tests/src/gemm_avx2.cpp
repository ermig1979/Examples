#include "gemm.h"

namespace Avx2
{
    void Micro32f6x16(int K, const float* A, int lda, int step,
        const float* B, int ldb, float* C, int ldc, int zero)
    {
        __m256 c00, c10, c20, c30, c40, c50, c01, c11, c21, c31, c41, c51, b0, b1, a0;
        if (zero)
        {
            c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
            c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
            c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
            c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
            c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
            c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
        }
        else
        {
            c00 = _mm256_loadu_ps(C + 0 * ldc + 0), c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
            c10 = _mm256_loadu_ps(C + 1 * ldc + 0), c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
            c20 = _mm256_loadu_ps(C + 2 * ldc + 0), c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
            c30 = _mm256_loadu_ps(C + 3 * ldc + 0), c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
            c40 = _mm256_loadu_ps(C + 4 * ldc + 0), c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
            c50 = _mm256_loadu_ps(C + 5 * ldc + 0), c51 = _mm256_loadu_ps(C + 5 * ldc + 8);
        }
        const int oa0 = lda * 0;
        const int oa1 = lda * 1;
        const int oa2 = lda * 2;
        const int oa3 = lda * 3;
        const int oa4 = lda * 4;
        const int oa5 = lda * 5;
        for (int k = 0; k < K; k++)
        {
            b0 = _mm256_loadu_ps(B + 0);
            b1 = _mm256_loadu_ps(B + 8);
            a0 = _mm256_set1_ps(A[oa0]), c00 = _mm256_fmadd_ps(a0, b0, c00), c01 = _mm256_fmadd_ps(a0, b1, c01);
            a0 = _mm256_set1_ps(A[oa1]), c10 = _mm256_fmadd_ps(a0, b0, c10), c11 = _mm256_fmadd_ps(a0, b1, c11);
            a0 = _mm256_set1_ps(A[oa2]), c20 = _mm256_fmadd_ps(a0, b0, c20), c21 = _mm256_fmadd_ps(a0, b1, c21);
            a0 = _mm256_set1_ps(A[oa3]), c30 = _mm256_fmadd_ps(a0, b0, c30), c31 = _mm256_fmadd_ps(a0, b1, c31);
            a0 = _mm256_set1_ps(A[oa4]), c40 = _mm256_fmadd_ps(a0, b0, c40), c41 = _mm256_fmadd_ps(a0, b1, c41);
            a0 = _mm256_set1_ps(A[oa5]), c50 = _mm256_fmadd_ps(a0, b0, c50), c51 = _mm256_fmadd_ps(a0, b1, c51);
            B += ldb; A += step;
        }
        _mm256_storeu_ps(C + 0 * ldc + 0, c00), _mm256_storeu_ps(C + 0 * ldc + 8, c01);
        _mm256_storeu_ps(C + 1 * ldc + 0, c10), _mm256_storeu_ps(C + 1 * ldc + 8, c11);
        _mm256_storeu_ps(C + 2 * ldc + 0, c20), _mm256_storeu_ps(C + 2 * ldc + 8, c21);
        _mm256_storeu_ps(C + 3 * ldc + 0, c30), _mm256_storeu_ps(C + 3 * ldc + 8, c31);
        _mm256_storeu_ps(C + 4 * ldc + 0, c40), _mm256_storeu_ps(C + 4 * ldc + 8, c41);
        _mm256_storeu_ps(C + 5 * ldc + 0, c50), _mm256_storeu_ps(C + 5 * ldc + 8, c51);
    }

    void Reorder32fA6(const float* A, int lda, int M, int K, float* bufA)
    {
        for (int i = 0; i < M; i += 6)
        {
            for (int k = 0; k < K; k += 4)
            {
                const float* pA = A + k;
                __m128 a0 = _mm_loadu_ps(pA + 0 * lda);
                __m128 a1 = _mm_loadu_ps(pA + 1 * lda);
                __m128 a2 = _mm_loadu_ps(pA + 2 * lda);
                __m128 a3 = _mm_loadu_ps(pA + 3 * lda);
                __m128 a4 = _mm_loadu_ps(pA + 4 * lda);
                __m128 a5 = _mm_loadu_ps(pA + 5 * lda);
                __m128 a00 = _mm_unpacklo_ps(a0, a2);
                __m128 a01 = _mm_unpacklo_ps(a1, a3);
                __m128 a10 = _mm_unpackhi_ps(a0, a2);
                __m128 a11 = _mm_unpackhi_ps(a1, a3);
                __m128 a20 = _mm_unpacklo_ps(a4, a5);
                __m128 a21 = _mm_unpackhi_ps(a4, a5);
                _mm_storeu_ps(bufA + 0, _mm_unpacklo_ps(a00, a01));
                _mm_storel_pi((__m64*)(bufA + 4), a20);
                _mm_storeu_ps(bufA + 6, _mm_unpackhi_ps(a00, a01));
                _mm_storeh_pi((__m64*)(bufA + 10), a20);
                _mm_storeu_ps(bufA + 12, _mm_unpacklo_ps(a10, a11));
                _mm_storel_pi((__m64*)(bufA + 16), a21);
                _mm_storeu_ps(bufA + 18, _mm_unpackhi_ps(a10, a11));
                _mm_storeh_pi((__m64*)(bufA + 22), a21);
                bufA += 24;
            }
            A += 6 * lda;
        }
    }

    void Reorder32fB16(int K, const float* B, int ldb, float* bufB)
    {
        for (int k = 0; k < K; ++k, B += ldb, bufB += 16)
        {
            _mm256_storeu_ps(bufB + 0, _mm256_loadu_ps(B + 0));
            _mm256_storeu_ps(bufB + 8, _mm256_loadu_ps(B + 8));
        }
    }

    void Macro32f(int M, int N, int K, const float* A, const float* B, 
        int ldb, float* bufB, bool reorderB, float* C, int ldc, int zero)
    {
        for (int j = 0; j < N; j += 16)
        {
            if (reorderB)
                Reorder32fB16(K, B + j, ldb, bufB + K * j);
            for (int i = 0; i < M; i += 6)
                Micro32f6x16(K, A + i * K, 1, 6, bufB + K * j, 16, C + i * ldc + j, ldc, zero);
        }
    }

    void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C)
    {
        const int L1 = 48 * 1024, L2 = 2 * 1024 * 1024, L3 = 32 * 1024 * 1024;
        int mK = std::min(L1 / 4 / 16, K) / 4 * 4;
        int mM = std::min(L2 / 4 / mK, M) / 6 * 6;
        int mN = std::min(L3 / 4 / mK, N) / 16 * 16;
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
                    Reorder32fA6(A + i * K + k, K, dM, dK, bufA.p);
                    Macro32f(dM, dN, dK, bufA.p, B + k * N + j, N, bufB.p, i == 0, C + i * N + j, N, k == 0);
                }
            }
        }
    }
}

