#include "defs.h"

#include <device_functions.h>

namespace a
{
	const int _K = 16;

	const int TSM = 16 * 8;
	const int TSN = 16 * _K;
	const int TSK = 32;
	const int WPT_M = 8;
	const int WPT_N = 8;
	const int RTS_M = TSM / WPT_M;
	const int RTS_N = TSN / WPT_N;
	const int TPAK = 4;
	const int TPBN = _K;

	__global__ void gemm_v8a(int M, int N, int K, const float4* A, int lda, const float4* B, int ldb, float* C, int ldc)
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
			{
				unsigned int Arow = block_id / TPAK;
				unsigned int Acol = (block_id % TPAK) * 8;
				unsigned int indexA = ((TSM * blockIdx.y + Arow) * lda + t * TSK + Acol) / 4;

#pragma unroll
				for (unsigned int vec = 0; vec < 2; vec++)
				{
#if __CUDA_ARCH__ >= 320
					float4 vecA = __ldg(&A[indexA + vec]);
#else
					float4 vecA = A[indexA + vec];
#endif
					As[Arow][Acol + vec * 4 + 0] = vecA.x;
					As[Arow][Acol + vec * 4 + 1] = vecA.y;
					As[Arow][Acol + vec * 4 + 2] = vecA.z;
					As[Arow][Acol + vec * 4 + 3] = vecA.w;
				}
			}

			{
				unsigned int Brow = block_id / TPBN;
				unsigned int Bcol = (block_id % TPBN) * 16;
				unsigned int indexB = ((TSK * t + Brow) * ldb + TSN * blockIdx.x + Bcol) / 4;
#pragma unroll
				for (unsigned int vec = 0; vec < 4; vec++)
				{
#if __CUDA_ARCH__ >= 320
					float4 vecB = __ldg(&B[indexB + vec]);
#else
					float4 vecB = B[indexB + vec];
#endif
					Bs[Brow][Bcol + vec * 4 + 0] = vecB.x;
					Bs[Brow][Bcol + vec * 4 + 1] = vecB.y;
					Bs[Brow][Bcol + vec * 4 + 2] = vecB.z;
					Bs[Brow][Bcol + vec * 4 + 3] = vecB.w;
				}
			}
			__syncthreads();

#pragma unroll
			for (unsigned int k = 0; k < TSK; k++)
			{
#pragma unroll
				for (int wn = 0; wn < WPT_N; wn++)
				{
					Breg[wn] = Bs[k][col + wn * RTS_N];
				}

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
}

int gemm_gpu_v8a(int M, int N, int K, const float* A, const float* B, float* C)
{
	using namespace a;
    dim3 grid(RTS_N, RTS_M);
    dim3 block(N / TSN, M / TSM);
	const int n = repeats(M, N, K, 0.180);
    for (int i = 0; i < n; ++i)
    {
		gemm_v8a <<<block, grid >>> (M, N, K, (const float4*)A, K, (const float4*)B, N, C, N);
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			printf("Error: %s\n", cudaGetErrorString(error));
			assert(error == cudaSuccess);
		}    
	}

    return n;
}

namespace a1
{
	const int TSM = 128;
	const int TSN = 128;
	const int TSK = 48;
	const int WPT_M = 8;
	const int WPT_N = 8;
	const int RTS_M = TSM / WPT_M;
	const int RTS_N = TSN / WPT_N;
	const int TPAK = 2;
	const int TPBN = 16;

	__global__ void gemm_v8a(int M, int N, int K, const float4* A, int lda, const float4* B, int ldb, float* C, int ldc)
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
			//if (t == 0)
			{
				unsigned int Arow = block_id / TPAK;
				unsigned int Acol = (block_id % TPAK) * 24;
				unsigned int indexA = ((TSM * blockIdx.y + Arow) * lda + t * TSK + Acol) / 4;

#pragma unroll
				for (unsigned int vec = 0; vec < 6; vec++)
				{
#if __CUDA_ARCH__ >= 320
					float4 vecA = __ldg(&A[indexA + vec]);
#else
					float4 vecA = A[indexA + vec];
#endif
					As[Arow][Acol + vec * 4 + 0] = vecA.x;
					As[Arow][Acol + vec * 4 + 1] = vecA.y;
					As[Arow][Acol + vec * 4 + 2] = vecA.z;
					As[Arow][Acol + vec * 4 + 3] = vecA.w;
				}

				unsigned int Brow = (block_id / TPBN) * 3;
				unsigned int Bcol = (block_id % TPBN) * 8;
				unsigned int indexB = ((TSK * t + Brow) * ldb + TSN * blockIdx.x + Bcol) / 4;

#pragma unroll
				for (unsigned int vec = 0; vec < 2; vec++)
				{
#if __CUDA_ARCH__ >= 320
					float4 vecB0 = __ldg(&B[indexB + ldb / 4 * 0 + vec]);
					float4 vecB1 = __ldg(&B[indexB + ldb / 4 * 1 + vec]);
					float4 vecB2 = __ldg(&B[indexB + ldb / 4 * 2 + vec]);
#else
					float4 vecB0 = B[indexB + ldb / 4 * 0 + vec];
					float4 vecB1 = B[indexB + ldb / 4 * 1 + vec];
					float4 vecB2 = B[indexB + ldb / 4 * 2 + vec];
#endif
					Bs[Brow + 0][Bcol + vec * 4 + 0] = vecB0.x;
					Bs[Brow + 0][Bcol + vec * 4 + 1] = vecB0.y;
					Bs[Brow + 0][Bcol + vec * 4 + 2] = vecB0.z;
					Bs[Brow + 0][Bcol + vec * 4 + 3] = vecB0.w;
					Bs[Brow + 1][Bcol + vec * 4 + 0] = vecB1.x;
					Bs[Brow + 1][Bcol + vec * 4 + 1] = vecB1.y;
					Bs[Brow + 1][Bcol + vec * 4 + 2] = vecB1.z;
					Bs[Brow + 1][Bcol + vec * 4 + 3] = vecB1.w;
					Bs[Brow + 2][Bcol + vec * 4 + 0] = vecB2.x;
					Bs[Brow + 2][Bcol + vec * 4 + 1] = vecB2.y;
					Bs[Brow + 2][Bcol + vec * 4 + 2] = vecB2.z;
					Bs[Brow + 2][Bcol + vec * 4 + 3] = vecB2.w;
				}
			}
			__syncthreads();

#pragma unroll
			for (unsigned int k = 0; k < TSK; k++)
			{
#pragma unroll
				for (int wn = 0; wn < WPT_N; wn++)
				{
					Breg[wn] = Bs[k][col + wn * RTS_N];
				}

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
}

namespace a2
{
	const int _K = 10;

	const int TSM = 16 * _K;
	const int TSN = 16 * _K;
	const int TSK = 32;
	const int WPT_M = _K;
	const int WPT_N = _K;
	const int RTS_M = TSM / WPT_M;
	const int RTS_N = TSN / WPT_N;
	const int TPAK = 8;
	const int TPBN = 8;

	__global__ void gemm_v8a(int M, int N, int K, const float4* A, int lda, const float4* B, int ldb, float* C, int ldc)
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
			//if (t == 0)
			{
				unsigned int Arow = (block_id / TPAK) * 5;
				unsigned int Acol = (block_id % TPAK) * 4;
				unsigned int Brow = block_id / TPBN;
				unsigned int Bcol = (block_id % TPBN) * 20;
				unsigned int indexA = ((TSM * blockIdx.y + Arow) * lda + t * TSK + Acol) / 4;
				unsigned int indexB = ((TSK * t + Brow) * ldb + TSN * blockIdx.x + Bcol) / 4;

#pragma unroll
				for (unsigned int vec = 0; vec < 5; vec++)
				{
#if __CUDA_ARCH__ >= 320
					float4 vecA = __ldg(&A[indexA + vec * lda / 4]);
					float4 vecB = __ldg(&B[indexB + vec]);
#else
					float4 vecA = A[indexA + vec * lda / 4];
					float4 vecB = B[indexB + vec];
#endif
					As[Arow + vec][Acol + 0] = vecA.x;
					As[Arow + vec][Acol + 1] = vecA.y;
					As[Arow + vec][Acol + 2] = vecA.z;
					As[Arow + vec][Acol + 3] = vecA.w;
					Bs[Brow][Bcol + vec * 4 + 0] = vecB.x;
					Bs[Brow][Bcol + vec * 4 + 1] = vecB.y;
					Bs[Brow][Bcol + vec * 4 + 2] = vecB.z;
					Bs[Brow][Bcol + vec * 4 + 3] = vecB.w;
				}
			}
			__syncthreads();

#pragma unroll
			for (unsigned int k = 0; k < TSK; k++)
			{
#pragma unroll
				for (int wn = 0; wn < WPT_N; wn++)
				{
					Breg[wn] = Bs[k][col + wn * RTS_N];
				}

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
}

