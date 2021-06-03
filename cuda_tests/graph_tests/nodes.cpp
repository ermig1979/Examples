#include "nodes.h"

void add_cpu(int size, const float* a, const float* b, float* c)
{
    for (int i = 0; i < size; ++i)
        c[i] = a[i] + b[i];
}

void add_cublas(cublasHandle_t handle, int size, const float* a, const float* b, float* c)
{
    cublasStatus_t status;
    status = cublasScopy_v2(handle, size, a, 1, c, 1);
    assert(status == CUBLAS_STATUS_SUCCESS);

    const float alpha = 1.0f;
    status = cublasSaxpy_v2(handle, size, &alpha, b, 1, c, 1);
    assert(status == CUBLAS_STATUS_SUCCESS);
}

