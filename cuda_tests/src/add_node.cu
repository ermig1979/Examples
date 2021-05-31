#include "nodes.h"

__global__ void add_kernel(size_t size, const float* a, const float* b, float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) 
    {
        c[i] = a[i] + b[i];
    }
}

void add_node(node_t* node)
{
    const gpu_tensor_t& a = node->src[0];
    const gpu_tensor_t& b = node->src[1];
    gpu_tensor_t& c = node->dst[0];

    add_kernel <<<2, (a.size + 1) / 2 >> > (a.size, a.data, b.data, c.data);
}