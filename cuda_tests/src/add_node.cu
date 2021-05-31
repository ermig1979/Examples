#include "nodes.h"

__global__ void add_kernel(int size, const float* a, const float* b, float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) 
    {
        c[i] = a[i] + b[i];
    }
}
