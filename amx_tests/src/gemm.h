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
}

//-------------------------------------------------------------------------------------------------

namespace Avx512bw
{
	void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C);
}

//-------------------------------------------------------------------------------------------------

namespace Amx
{
	void InitAmx();

	void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C);
}

//-------------------------------------------------------------------------------------------------

inline void Gemm32f(const Mat32f& a, const Mat32f& b, Mat32f& c, const Gemm32fPtr gemm)
{
	assert(a.m == c.m && a.n == b.m && b.n == c.n);
	gemm(a.m, b.n, a.n, a.p, b.p, c.p);
}

