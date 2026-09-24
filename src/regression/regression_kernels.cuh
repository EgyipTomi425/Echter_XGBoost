#pragma once

#include "regression_kernels.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace echter::xgb::detail
{

__global__ void regression_predict_kernel(
    const float* __restrict__ device_features,
    float* __restrict__ device_output,
    std::size_t rows,
    const Node* __restrict__ nodes,
    const int* __restrict__ entry_nodes,
    int tree_count,
    float base_score);

}
