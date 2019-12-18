#include "defs.h"

#include <device_functions.h>

const int TSM = 128;
const int TSN = 128;
const int TSK = 32;
const int WPT_M = 8;                       
const int WPT_N = 8;
const int RTS_M = TSM / WPT_M;              
const int RTS_N = TSN / WPT_N;              
const int TPAK = 2;    // How many thread can read a row of tile A, == TSK / (VPTAK * 4)
const int TPBN = 8;    // How many thread can read a row of tile B, == TSN / (VPTBN * 4)

__global__ void gemm_v7a(int M, int N, int K, const float4 * A, int lda, const float4 * B, int ldb, float* C, int ldc)
{
	const unsigned int col = threadIdx.x; 
	const unsigned int row = threadIdx.y; 
	const unsigned int globalRow = TSM * blockIdx.y + row; 
	const unsigned int globalCol = TSN * blockIdx.x + col; 
	const unsigned int block_id = row * blockDim.x + col;

	__shared__ float As[TSM][TSK];
	__shared__ float Bs[TSK][TSN];

	float Cres[WPT_M][WPT_N];
	float Breg[WPT_N];
#pragma unroll
	for (unsigned int wm = 0; wm < WPT_M; wm++)
#pragma unroll	
		for (unsigned int wn = 0; wn < WPT_N; wn++)
			Cres[wm][wn] = 0;

	const unsigned int numTiles = K / TSK;
	for (unsigned int t = 0; t < numTiles; t++)
	{
		unsigned int Arow = block_id / TPAK;
		unsigned int Acol = (block_id % TPAK) * 16;
		unsigned int Brow = block_id / TPBN;
		unsigned int Bcol = (block_id % TPBN) * 16;
		unsigned int indexA = ((TSM * blockIdx.y + Arow) * lda + t * TSK + Acol) / 4;
		unsigned int indexB = ((TSK * t + Brow) * ldb + TSN * blockIdx.x + Bcol) / 4;
#pragma unroll
		for (unsigned int vec = 0; vec < 4; vec++)
		{
#if __CUDA_ARCH__ >= 320
			float4 vecA = __ldg(&A[indexA + vec]);
			float4 vecB = __ldg(&B[indexB + vec]);
#else
			float4 vecA = A[indexA + vec];
			float4 vecB = B[indexB + vec];
#endif
			As[Arow][Acol + vec * 4 + 0] = vecA.x;
			As[Arow][Acol + vec * 4 + 1] = vecA.y;
			As[Arow][Acol + vec * 4 + 2] = vecA.z;
			As[Arow][Acol + vec * 4 + 3] = vecA.w;
			Bs[Brow][Bcol + vec * 4 + 0] = vecB.x;
			Bs[Brow][Bcol + vec * 4 + 1] = vecB.y;
			Bs[Brow][Bcol + vec * 4 + 2] = vecB.z;
			Bs[Brow][Bcol + vec * 4 + 3] = vecB.w;
		}
		__syncthreads();

#pragma unroll
		for (unsigned int k = 0; k < TSK; k++)
		{
#pragma unroll
			for (int wn = 0; wn < WPT_N; wn++)
				Breg[wn] = Bs[k][col + wn * RTS_N];

#pragma unroll
			for (unsigned int wm = 0; wm < WPT_M; wm++)
			{
				float Areg_wm = As[row + wm * RTS_M][k];
#pragma unroll
				for (unsigned int wn = 0; wn < WPT_N; wn++)
				{
					Cres[wm][wn] += Areg_wm * Breg[wn];
				}
			}
		}
		__syncthreads();
	}

#pragma unroll
	for (unsigned int wm = 0; wm < WPT_M; wm++)
	{
		unsigned int c_dim1 = (globalRow + wm * RTS_M) * ldc;
#pragma unroll
		for (unsigned int wn = 0; wn < WPT_N; wn++)
		{
			unsigned int c_coord = globalCol + wn * RTS_N + c_dim1;
			C[c_coord] = Cres[wm][wn];
		}
	}
}

int gemm_gpu_v7a(int M, int N, int K, const float* A, const float* B, float* C)
{
    dim3 grid(RTS_M, RTS_N);
    dim3 block(M / TSM, N / TSN);
    const int n = repeats(M, N, K, 0.180);
    for (int i = 0; i < n; ++i)
    {
		gemm_v7a <<<block, grid >>> (M, N, K, (const float4*)A, K, (const float4*)B, N, C, N);
    }
    assert(cudaGetLastError() == cudaSuccess);
    return n;
}
