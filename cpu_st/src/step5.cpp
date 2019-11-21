#include "defs.h"

void macro_v5(int M, int N, int K, const float * A, int lda, 
    const float * B, int ldb, float * bufB, float * C, int ldc)
{
    for (int j = 0; j < N; j += 16)
    {
        reorder_b_16(K, B + j, ldb, bufB);
        for (int i = 0; i < M; i += 6)
            micro_6x16(K, A + i*lda, lda, 1, bufB, 16, C + i*ldc + j, ldc);
    }
}

void gemm_v5(int M, int N, int K, const float * A, const float * B, float * C)
{
    const int L1 = 32 * 1024;
    int mK = std::min(L1 / 4 / 16, K);
    buf_t bufB(16 * mK);
    for(int k = 0; k < K; k += mK)
    {
        int dK = std::min(K, k + mK) - k;
        if(k == 0)
            init_c(M, N, C, N);
        macro_v5(M, N, dK, A + k, K, B + k*N, N, bufB.p, C, N);
    }
}