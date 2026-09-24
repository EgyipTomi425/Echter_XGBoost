#include "regression_kernels.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace echter::xgb::detail
{

namespace
{

__global__ void regression_predict_kernel(
    const float* __restrict__ device_features,
    float* __restrict__ device_output,
    std::size_t rows,
    const Node* __restrict__ nodes,
    const int* __restrict__ entry_nodes,
    int tree_count,
    float base_score)
{
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= rows)
    {
        return;
    }

    float sum = 0.0f;

    for (int tree = 0; tree < tree_count; ++tree)
    {
        Node node = nodes[entry_nodes[tree]];

        while (node.left >= 0)
        {
            const float value = device_features[
                static_cast<std::size_t>(node.split_feature & ~default_left_flag) * rows + row];

            const bool go_left = isnan(value)
                ? (node.split_feature & default_left_flag) != 0
                : value < node.value;

            node = nodes[go_left ? node.left : node.right];
        }

        sum += node.value;
    }

    device_output[row] = base_score + sum;
}

}

void launch_regression_kernel(
    const float* device_features,
    float* device_output,
    std::size_t rows,
    const DeviceModel& model)
{
    constexpr unsigned int threads_per_block = 256;
    const auto blocks = static_cast<unsigned int>(
        (rows + threads_per_block - 1) / threads_per_block);

    regression_predict_kernel<<<blocks, threads_per_block>>>(
        device_features,
        device_output,
        rows,
        model.nodes.get(),
        model.entry_nodes.get(),
        model.num_trees,
        model.base_score);
}

}
