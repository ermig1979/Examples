#include "defs.h"

#include <device_functions.h>

const int TSM = 128;
const int TSN = 128;
const int TSK = 16;
const int WPTM = 8;                       
const int WPTN = 8;
const int RTSM = TSM / WPTM;              
const int RTSN = TSN / WPTN;              
const int LPTA = TSK * WPTM * WPTN / TSN; 
//const int LPTB = TSK * WPTM * WPTN / TSM;

__global__ void gemm_v6a(int M, int N, int K, const float* A, const float* B, float* C)
{
    const int tidm = threadIdx.y; 
    const int tidn = threadIdx.x; 
    const int offsetM = TSM * blockIdx.y;
    const int offsetN = TSN * blockIdx.x;

    __shared__ float sA[TSK][TSM];
    __shared__ float sB[TSN][TSK];

    float rA;
    float rB[WPTN];
    float rC[WPTM][WPTN];

#pragma unroll
    for (int wm = 0; wm < WPTM; wm++) 
    {
#pragma unroll
        for (int wn = 0; wn < WPTN; wn++)
            rC[wm][wn] = 0.0f;
    }

    for( int k0 = 0; k0 < K; k0 += TSK)
    {
#pragma unroll
        for (int la = 0; la < LPTA; la++)
        {
            int tid = tidn * RTSM + tidm;
            int id = la * RTSN * RTSM + tid;
            int row = id % TSM;
            int col = id / TSM;
            int tiledIndex = k0 + col;
#if __CUDA_ARCH__ >= 320
            sA[col][row] = __ldg(&A[tiledIndex * M + offsetM + row]);
            sB[row][col] = __ldg(&B[tiledIndex * N + offsetN + row]);
#else
            sA[col][row] = A[tiledIndex * M + offsetM + row];
            sB[row][col] = B[tiledIndex * N + offsetN + row];
#endif
        }
        __syncthreads();
        for (int k = 0; k < TSK; k++) 
        {
#pragma unroll
            for (int wn = 0; wn < WPTN; wn++) 
            {
                int col = tidn + wn * RTSN;
                rB[wn] = sB[col][k];
            }
#pragma unroll
            for (int wm = 0; wm < WPTM; wm++)
            {
                int row = tidm + wm * RTSM;
                rA = sA[k][row];
#pragma unroll
                for (int wn = 0; wn < WPTN; wn++) {
                    rC[wm][wn] += rA * rB[wn];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int wm = 0; wm < WPTM; wm++) 
    {
        int globalRow = offsetM + tidm + wm * RTSM;
#pragma unroll
        for (int wn = 0; wn < WPTN; wn++) 
        {
            int globalCol = offsetN + tidn + wn * RTSN;
            C[globalCol + globalRow * N] = rC[wm][wn];
        }
    }
}

int gemm_gpu_v6a(int M, int N, int K, const float* A, const float* B, float* C)
{
    gpu_buf_t tA(M * K);
    dim3 gridT(TRX, TRY);
    dim3 blockT((M + TRX - 1) / TRX, (K + TRY - 1) / TRY);

    dim3 grid(TSM / WPTM, TSN / WPTN);
    dim3 block(M / TSM, N / TSN);
    const int n = repeats(M, N, K, 0.370);
    for (int i = 0; i < n; ++i)
    {
        transpose<<<blockT, gridT>>>(M, K, A, tA.p);
        gemm_v6a<<<block, grid>>>(M, N, K, tA.p, B, C);
    }
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}
