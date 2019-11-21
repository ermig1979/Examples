#include "defs.h"

void reorder_a_6(const float * A, int lda, int M, int K, float * bufA)
{
    for (int i = 0; i < M; i += 6)
    {
        for (int k = 0; k < K; k += 4)
        {
            const float * pA = A + k;
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

void macro_v6(int M, int N, int K, const float * A, 
    const float * B, int ldb, float * bufB, float * C, int ldc)
{
    for (int j = 0; j < N; j += 16)
    {
        reorder_b_16(K, B + j, ldb, bufB);
        for (int i = 0; i < M; i += 6)
            micro_6x16(K, A + i*K, 1, 6, bufB, 16, C + i*ldc + j, ldc);
    }
}

void gemm_v6(int M, int N, int K, const float * A, const float * B, float * C)
{
    const int L1 = 32 * 1024, L2 = 256*1024;
    int mK = std::min(L1 / 4 / 16, K) / 4 * 4;
    int mM = std::min(L2 / 4 / mK, M) / 6 * 6;
    buf_t bufB(16 * mK);
    buf_t bufA(mK * mM);
    for(int k = 0; k < K; k += mK)
    {
        int dK = std::min(K, k + mK) - k;
        for (int i = 0; i < M; i += mM)
        {
            int dM = std::min(M, i + mM) - i;
            if (k == 0)
                init_c(dM, N, C + i * N, N);
            reorder_a_6(A + i * K + k, K, dM, dK, bufA.p);
            macro_v6(dM, N, dK, bufA.p, B + k * N, N, bufB.p, C + i * N, N);
        }
    }
}