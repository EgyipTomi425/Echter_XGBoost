#include "regression_kernels.cuh"

namespace echter::xgb::detail
{

__global__ void regression_predict_kernel(
    const float* __restrict__ device_features,
    float* __restrict__ device_output,
    std::size_t rows,
    const Node* __restrict__ nodes,
    const int* __restrict__ entry_nodes,
    int tree_count,
    float base_score,
    std::uint32_t feature_bits)
{
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= rows)
    {
        return;
    }

    const std::uint32_t feature_mask = (1u << feature_bits) - 1u;
    const std::uint32_t offset_shift = feature_bits + 1u;
    float sum = 0.0f;

    for (int tree = 0; tree < tree_count; ++tree)
    {
        int index = entry_nodes[tree];
        Node node = nodes[index];

        while (node.link != 0)
        {
            const float value = device_features[
                static_cast<std::size_t>((node.link >> 1) & feature_mask) * rows + row];

            const bool go_right = isnan(value)
                ? (node.link & 1u) == 0
                : !(value < node.value);

            index += static_cast<int>(node.link >> offset_shift) + go_right;
            node = nodes[index];
        }

        sum += node.value;
    }

    device_output[row] = base_score + sum;
}

void launch_regression_kernel(
    const float* device_features,
    float* device_output,
    std::size_t rows,
    const DeviceModel& model)
{
    constexpr unsigned int threads_per_block = 1024;
    const auto blocks = static_cast<unsigned int>(
        (rows + threads_per_block - 1) / threads_per_block);

    regression_predict_kernel<<<blocks, threads_per_block>>>(
        device_features,
        device_output,
        rows,
        model.nodes.get(),
        model.entry_nodes.get(),
        model.num_trees,
        model.base_score,
        model.feature_bits);
}

}
