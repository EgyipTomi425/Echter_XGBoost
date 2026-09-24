#include "model_backend.hpp"

#include "cuda_utils.cuh"

namespace echter::xgb::model_backend
{

void* allocate_device(std::size_t rows, std::size_t features)
{
    return new detail::DeviceColumnarBuffer(rows, features);
}

void destroy_device(void* device) noexcept
{
    delete static_cast<detail::DeviceColumnarBuffer*>(device);
}

DeviceView view(const void* device) noexcept
{
    if (device == nullptr)
    {
        return {};
    }

    const auto& buffer = *static_cast<const detail::DeviceColumnarBuffer*>(device);
    return {buffer.data.get(), buffer.rows, buffer.features};
}

void upload(const float* host, void* device)
{
    auto& buffer = *static_cast<detail::DeviceColumnarBuffer*>(device);
    const std::size_t elements = buffer.rows * buffer.features;
    if (elements == 0)
    {
        return;
    }

    detail::check_cuda(
        cudaMemcpy(
            buffer.data.get(),
            host,
            elements * sizeof(float),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(columnar input H2D)");
}

void download(DeviceView device, float* host)
{
    const std::size_t elements = device.rows * device.features;
    if (elements == 0)
    {
        return;
    }

    detail::check_cuda(
        cudaMemcpy(
            host,
            device.data,
            elements * sizeof(float),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(columnar output D2H)");
}

}
