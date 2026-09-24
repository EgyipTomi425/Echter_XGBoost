#include "model_backend.hpp"

#include "../regression/regression_internal.cuh"

namespace echter::xgb::model_backend
{

void* allocate_device(std::size_t rows, std::size_t features)
{
    auto* buffer = new detail::DeviceColumnarBuffer{};
    if (!detail::allocate_columnar(rows, features, *buffer))
    {
        delete buffer;
        return nullptr;
    }
    return buffer;
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
    return {buffer.data, buffer.rows, buffer.features};
}

bool upload(const float* host, std::size_t rows, std::size_t features, void* device)
{
    if (device == nullptr)
    {
        return false;
    }

    return detail::upload_columnar(
        host, rows, features,
        *static_cast<detail::DeviceColumnarBuffer*>(device));
}

bool download(DeviceView device, float* host, std::size_t elements)
{
    if ((elements != 0 && (device.data == nullptr || host == nullptr)))
    {
        return false;
    }

    return cudaMemcpy(
        host,
        device.data,
        elements * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool select_columns(DeviceView source, std::size_t excluded, void* destination)
{
    if (destination == nullptr || excluded >= source.features)
    {
        return false;
    }

    const auto target = static_cast<detail::DeviceColumnarBuffer*>(destination);
    if (target->rows != source.rows || target->features + 1 != source.features)
    {
        return false;
    }

    std::size_t target_column = 0;
    for (std::size_t source_column = 0; source_column < source.features; ++source_column)
    {
        if (source_column == excluded)
        {
            continue;
        }

        if (cudaMemcpy(
                target->data + target_column * source.rows,
                source.data + source_column * source.rows,
                source.rows * sizeof(float),
                cudaMemcpyDeviceToDevice) != cudaSuccess)
        {
            return false;
        }
        ++target_column;
    }
    return true;
}

}
