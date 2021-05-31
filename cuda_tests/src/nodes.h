#pragma once

#include "tensor.h"

__global__ void add_kernel(int size, const float* a, const float* b, float* c);

void* add_kernel_ptr();
