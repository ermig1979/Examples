#include "nodes.h"

__global__ void add_kernel(int size, const float* a, const float* b, float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) 
    {
        c[i] = a[i] + b[i];
    }
}

//__global__ void add_kernel_cublas(cublasHandle_t handle, int size, const float* a, const float* b, float* c)
//{
//    //ublasHandle_t handle;
//    //cublasCreate(&handle);
//
//    cublasScopy(handle, size, a, 1, c, 1);
//
//    const float alpha = 1.0f;
//    cublasSaxpy(handle, size, &alpha, b, 1, c, 1);
////
//    //cublasDestroy_v2(handle);
//}
