#pragma once

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <cuda_runtime.h>

inline bool check(cudaError_t error, const std::string & action, const std::string& file, int line)
{
    if (error == cudaSuccess)
        return true;
    else
    {
        std::cout << action << " return error: " << cudaGetErrorName(error);
        std::cout << ", " << file << ", " << line<< std::endl;
        assert(0);
        return false;
    }
}

#define CHECK(action) check(action, #action, __FILE__, __LINE__)

inline int repeats(int M, int N, int K, double k)
{
    int n = std::max(int(double(1024) / double(M) * double(1024) / double(N) * double(1024) / double(K) * k * 1000.0), 1);
    //std::cout << "repeats = " << n << std::endl;;
    return n;
}

typedef int (*gemm_t)(int M, int N, int K, const float * A, const float * B, float * C);

int gemm_cublas(int M, int N, int K, const float* A, const float* B, float* C);
int gemm_gpu_v7a(int M, int N, int K, const float* A, const float* B, float* C);

const int TRX = 16;
const int TRY = 16;

