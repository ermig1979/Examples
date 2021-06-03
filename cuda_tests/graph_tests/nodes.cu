#include "nodes.h"

__global__ void add_kernel(int size, const float* a, const float* b, float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) 
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void scale_kernel(int size, float scale, float* data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        data[i] *= scale;
    }
}

