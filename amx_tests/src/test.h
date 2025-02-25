#pragma once

#include "mat.h"

typedef void (*Gemm32fPtr)(int M, int N, int K, const float* A, const float* B, float* C);
typedef void (*Gemm32f16bPtr)(int M, int N, int K, const float* A, const uint16_t* B, float* C);
typedef void (*Gemm16bPtr)(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C);
typedef void (*ConvertBPtr)(int N, int K, int microN, const float* src, uint16_t* dst);

//-------------------------------------------------------------------------------------------------

namespace Base
{
	void Gemm32f(int M, int N, int K, const float* A, const float* B, float* C);
	void Gemm16b(int M, int N, int K, const float* A, const float* B, float* C);
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

	void GemmFunc(int M, int N, int K, const float* A, const float* B, float* C);

    void ConvertB(int N, int K, int microN, const float* src, uint16_t* dst);
    void Gemm32f16b(int M, int N, int K, const float* A, const uint16_t* B, float* C);
	void Gemm32f16bV2(int M, int N, int K, const float* A, const uint16_t* B, float* C);
	void Gemm32f16bV3(int M, int N, int K, const float* A, const uint16_t* B, float* C);

	void ConvertA(int M, int K, const float* src, uint16_t* dst);
	void ReorderA(int M, int K, const float* src, uint16_t* dst);
	void Gemm16b(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C);

	void ConvertBV2(int N, int K, int microN, const float* src, uint16_t* dst);
	void Gemm16bV2(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C);
	void Gemm16bV3(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C);
	void Gemm16bV4(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C);
	void Gemm16bV5(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C);
}

//-------------------------------------------------------------------------------------------------

bool TestGemm(int M, int N, int K);


