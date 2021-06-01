#include "nodes.h"

void add_cpu(int size, const float* a, const float* b, float* c)
{
    for (int i = 0; i < size; ++i)
        c[i] = a[i] + b[i];
}

