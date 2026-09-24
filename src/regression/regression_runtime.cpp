#include "regression_kernels.hpp"
#include "regression_model.hpp"

#include "../core/cuda_check.cuh"

#include <cuda_runtime.h>

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
        "cudaMemcpy(model nodes)");
    check_cuda(
        cudaMemcpy(
            device.entry_nodes.get(),
            host.entry_nodes.data(),
            host.entry_nodes.size() * sizeof(int),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(model entry nodes)");
    return device;
}

DeviceColumnarBuffer predict(const DeviceModel& model, DeviceView input)
{
    DeviceColumnarBuffer output(input.rows, 1);
    if (input.rows == 0)
    {
        return output;
    }

    require_device_accessible(input.data);
    launch_regression_kernel(input.data, output.data.get(), input.rows, model);
    check_cuda(cudaGetLastError(), "regression kernel launch");
    check_cuda(cudaDeviceSynchronize(), "regression kernel");
    return output;
}

}
