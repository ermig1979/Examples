#include "defs.h"

void reorder_b_16(int K, const float * B, int ldb, float * bufB)
{
    for (int k = 0; k < K; ++k, B += ldb, bufB += 16)
    {
        _mm256_storeu_ps(bufB + 0, _mm256_loadu_ps(B + 0));
        _mm256_storeu_ps(bufB + 8, _mm256_loadu_ps(B + 8));
    }
}

void gemm_v4(int M, int N, int K, const float * A, const float * B, float * C)
{
    for (int j = 0; j < N; j += 16)
    {
        buf_t bufB(16*K);
        reorder_b_16(K, B + j, N, bufB.p);
        for (int i = 0; i < M; i += 6)
        {
            init_c(6, 16, C + i*N + j, N);
            micro_6x16(K, A + i*K, K, 1, bufB.p, 16, C + i*N + j, N);
        }
    }
}