#pragma once

#include "mat.h"

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
	void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C);

	void StubMicro16b(int M, int N, int K, const float* A, const float* B, float* C);
	void StubMacro16b(int M, int N, int K, const float* A, const float* B, float* C);
}

//-------------------------------------------------------------------------------------------------

inline void Gemm32f(const Mat32f& a, const Mat32f& b, Mat32f& c, const Gemm32fPtr gemm)
{
	assert(a.m == c.m && a.n == b.m && b.n == c.n);
	gemm(a.m, b.n, a.n, a.p, b.p, c.p);
}

void TestGemm32f(int M, int N, int K, const std::string& desc, Gemm32fPtr gemm, Gemm32fPtr control, double time = 1.0);

#define TEST_GEMM32F(M, N, K, gemm, control) TestGemm32f(M, N, K, #gemm, gemm, control)

bool TestGemm(int M, int N, int K);

