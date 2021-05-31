#pragma once

#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

inline bool check(cudaError_t error, const std::string& action, const std::string& file, int line)
{
    if (error == cudaSuccess)
        return true;
    else
    {
        std::cout << action << " return error: " << cudaGetErrorName(error);
        std::cout << ", " << file << ", " << line << std::endl;
        assert(0);
        return false;
    }
}

#define CHECK(action) check(action, #action, __FILE__, __LINE__)

