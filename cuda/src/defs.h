#pragma once

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <cuda_runtime.h>

inline int repeats(int M, int N, int K, double k)
{
    return int(double(1024) /double(M) * double(1024) / double(N) * double(1024) / double(K) * k * 1000.0);
}

typedef int (*gemm_t)(int M, int N, int K, const float * A, const float * B, float * C);

int gemm_cublas(int M, int N, int K, const float * A, const float * B, float * C);
int gemm_gpu_v0(int M, int N, int K, const float * A, const float * B, float * C);
int gemm_gpu_v1(int M, int N, int K, const float * A, const float * B, float * C);
int gemm_gpu_v2(int M, int N, int K, const float * A, const float * B, float * C);
//int gemm_gpu_v3(int M, int N, int K, const float * A, const float * B, float * C);
