#pragma once

#include "tensor.h"

void print_device_info();

void print_graph_info(cudaGraph_t graph);

bool check(const cpu_tensor_t& control, const cpu_tensor_t& current, const std::string& desc, float eps = 0.001f, int count = 32);


