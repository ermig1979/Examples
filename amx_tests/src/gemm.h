#pragma once

#include "mat.h"

typedef void (*Gemm32fPtr)(int M, int N, int K, const float* A, const float* B, float* C);
typedef void (*Gemm32f16bPtr)(int M, int N, int K, const float* A, const uint16_t* B, float* C);
typedef void (*Gemm16bPtr)(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C);

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

    void ReorderB(int N, int K, int microN, const float* src, uint16_t* dst);
    void Gemm32f16b(int M, int N, int K, const float* A, const uint16_t* B, float* C);
	void Gemm32f16bV2(int M, int N, int K, const float* A, const uint16_t* B, float* C);

	void ConvertA(int M, int K, const float* src, uint16_t* dst);
	void ReorderA(int M, int K, const float* src, uint16_t* dst);
	void Gemm16b(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C);

	void StubMicro16b(int M, int N, int K, const float* A, const float* B, float* C);
	void StubMacro16b(int M, int N, int K, const float* A, const float* B, float* C);
}

//-------------------------------------------------------------------------------------------------

bool TestGemm(int M, int N, int K);

