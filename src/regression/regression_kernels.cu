#include "regression_kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace echter::xgb::detail
{

namespace
{

// Split feature indexes and child links are validated when the model is
// loaded, and the input feature count is checked before launch, so the
// traversal needs no bounds checks.
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

    float sum = base_score;

    for (int tree = 0; tree < tree_count; ++tree)
    {
        int node_index = entry_nodes[tree];

        while (true)
        {
            const Node node = nodes[node_index];

            if (node.is_leaf != 0)
            {
                sum += node.leaf;
                break;
            }

            const float value = device_features[
                static_cast<std::size_t>(node.feature) * rows + row];

            const bool missing = isnan(value);
            const bool go_left = missing
                ? node.default_left != 0
                : value < node.threshold;

            node_index = go_left ? node.left : node.right;
        }
    }

    device_output[row] = sum;
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
