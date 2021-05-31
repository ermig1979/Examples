#pragma once

#include "tensor.h"

struct node_t
{
    gpu_tensors_t src, dst;
    cudaGraphNode_t graphNode;
};

void add_node(node_t* node);
