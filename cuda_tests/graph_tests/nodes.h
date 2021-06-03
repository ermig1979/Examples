#pragma once

#include "tensor.h"

__global__ void add_kernel(int size, const float* a, const float* b, float* c);

void add_cpu(int size, const float* a, const float* b, float* c);

void add_cublas(cublasHandle_t handle, int size, const float* a, const float* b, float* c);

__global__ void scale_kernel(int size, float scale, float* data);

void scale_cpu(int size, float scale, float* data);

void scale_cublas(cublasHandle_t handle, int size, float scale, float* data);

