#pragma once

#include "tensor.h"

void print_device_info();

bool check(const cpu_tensor_t& control, const cpu_tensor_t& current, const std::string& desc, float eps = 0.001f, int count = 32);


