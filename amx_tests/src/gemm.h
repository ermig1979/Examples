#pragma once

#include "mat.h"

//typedef void (*Gemm32fPtr)(const Mat32f& a, const Mat32f& b, Mat32f& c);
//typedef void (*Gemm16bPtr)(const Mat16b& a, const Mat16b& b, Mat32f& c);
//void Gemm32fV0(const Mat32f& a, const Mat32f& b, Mat32f& c);
//void Gemm32fV1(const Mat32f& a, const Mat32f& b, Mat32f& c);

typedef void (*Gemm32fPtr)(int M, int N, int K, const float* A, const float* B, float* C);

//-------------------------------------------------------------------------------------------------

namespace Base
{
	void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C);
	void Gemm16b(int M, int N, int K, const float* A, const float* B, float* C);
}

//-------------------------------------------------------------------------------------------------

namespace Avx2
{
	void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C);

	void Micro32f6x16(int K, const float* A, int lda, int step, const float* B, int ldb, float* C, int ldc);
	void InitC(int M, int N, float* C, int ldc);
	void Reorder32fB16(int K, const float* B, int ldb, float* _B);
	void Reorder32fA6(const float* A, int lda, int M, int K, float* bufA);
}

//-------------------------------------------------------------------------------------------------

namespace Avx512bw
{
	void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C);

	void Micro32f12x32(int K, const float* A, int lda, int step, const float* B, int ldb, float* C, int ldc);
	void InitC(int M, int N, float* C, int ldc);
	void Reorder32fB32(int K, const float* B, int ldb, float* _B);
	void Reorder32fA12(const float* A, int lda, int M, int K, float* bufA);
}

//-------------------------------------------------------------------------------------------------

inline void Gemm32f(const Mat32f& a, const Mat32f& b, Mat32f& c, const Gemm32fPtr gemm)
{
	assert(a.m == c.m && a.n == b.m && b.n == c.n);
	gemm(a.m, b.n, a.n, a.p, b.p, c.p);
}

