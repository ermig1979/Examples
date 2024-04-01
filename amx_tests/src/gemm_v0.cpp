#include "gemm.h"
#include "bf16.h"

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

    //-------------------------------------------------------------------------------------------------

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


