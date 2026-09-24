#include "regression_backend.hpp"
#include "regression_kernels.cuh"
#include "regression_model.cuh"

#include "../core/cuda_utils.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace echter::xgb::detail
{

DeviceModel upload_model(const HostModel& host)
{
    DeviceModel device;
    device.nodes = allocate_device_array<Node>(host.nodes.size());
    device.entry_nodes = allocate_device_array<int>(host.entry_nodes.size());
    device.num_trees = static_cast<int>(host.entry_nodes.size());
    device.num_features = host.num_features;
    device.base_score = host.base_score;

    check_cuda(
        cudaMemcpy(
            device.nodes.get(),
            host.nodes.data(),
            host.nodes.size() * sizeof(Node),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(model nodes H2D)");
    check_cuda(
        cudaMemcpy(
            device.entry_nodes.get(),
            host.entry_nodes.data(),
            host.entry_nodes.size() * sizeof(int),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(model entry nodes H2D)");

    return device;
}

}

namespace echter::xgb::regression_backend
{

void* create_model()
{
    return new detail::DeviceModel{};
}

void destroy_model(void* model) noexcept
{
    delete static_cast<detail::DeviceModel*>(model);
}

void load_model(void* model, const std::string& path)
{
    // The previously loaded model is replaced only after the new one has been
    // parsed and uploaded successfully.
    *static_cast<detail::DeviceModel*>(model) =
        detail::upload_model(detail::load_model_with_cudf(path));
}

int model_features(const void* model) noexcept
{
    return model == nullptr
        ? 0
        : static_cast<const detail::DeviceModel*>(model)->num_features;
}

int model_trees(const void* model) noexcept
{
    return model == nullptr
        ? 0
        : static_cast<const detail::DeviceModel*>(model)->num_trees;
}

void predict_device(
    model_backend::DeviceView device,
    const void* model,
    void* output)
{
    auto& result = *static_cast<detail::DeviceColumnarBuffer*>(output);
    if (result.rows != device.rows || result.features != 1)
    {
        throw std::invalid_argument("prediction output buffer has the wrong shape");
    }

    if (device.rows == 0)
    {
        return;
    }

    detail::launch_regression_kernel(
        device.data,
        result.data.get(),
        device.rows,
        *static_cast<const detail::DeviceModel*>(model));
    detail::check_cuda(cudaGetLastError(), "regression kernel launch");
    detail::check_cuda(cudaDeviceSynchronize(), "regression kernel");
}

}
