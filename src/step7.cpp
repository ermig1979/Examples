#include "defs.h"

void macro_v7(int M, int N, int K, const float * A, 
    const float * B, int ldb, float * bufB, bool reorderB, float * C, int ldc)
{
    for (int j = 0; j < N; j += 16)
    {
        if(reorderB)
            reorder_b_16(K, B + j, ldb, bufB + K*j);
        for (int i = 0; i < M; i += 6)
            micro_6x16(K, A + i*K, 1, 6, bufB + K*j, 16, C + i*ldc + j, ldc);
    }
}

void gemm_v7(int M, int N, int K, const float * A, const float * B, float * C)
{
    const int L1 = 32 * 1024, L2 = 256*1024, L3 = 2*1024*1024;
    int mK = std::min(L1 / 4 / 16, K) / 4 * 4;
    int mM = std::min(L2 / 4 / mK, M) / 6 * 6;
    int mN = std::min(L3 / 4 / mK, N) / 16 * 16;
    buf_t bufB(mN * mK);
    buf_t bufA(mK * mM);
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
                    init_c(dM, dN, C + i * N + j, N);
                reorder_a_6(A + i * K + k, K, dM, dK, bufA.p);
                macro_v7(dM, dN, dK, bufA.p, B + k * N + j, N, bufB.p, i == 0, C + i * N + j, N);
            }
        }
    }
}