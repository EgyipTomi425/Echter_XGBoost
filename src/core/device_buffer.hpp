#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace echter::xgb::detail
{

struct DeviceFree
{
    void operator()(void* pointer) const noexcept;
};

template <typename T>
using DeviceArray = std::unique_ptr<T[], DeviceFree>;

void* allocate_device_bytes(std::size_t bytes);

template <typename T>
DeviceArray<T> allocate_device_array(std::size_t count)
{
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
        throw std::length_error("device allocation size overflows");
    }
    return DeviceArray<T>(static_cast<T*>(allocate_device_bytes(count * sizeof(T))));
}

struct DeviceView
{
    const float* data{nullptr};
    std::size_t rows{0};
    std::size_t features{0};
};

struct DeviceColumnarBuffer
{
    DeviceColumnarBuffer() = default;

    DeviceColumnarBuffer(std::size_t rows, std::size_t features)
        : rows(rows),
          features(features)
    {
        if (rows != 0 && features > std::numeric_limits<std::size_t>::max() / rows)
        {
            throw std::length_error("columnar buffer size overflows");
        }
        data = allocate_device_array<float>(rows * features);
    }

    DeviceColumnarBuffer(DeviceColumnarBuffer&& other) noexcept
        : data(std::move(other.data)),
          rows(std::exchange(other.rows, 0)),
          features(std::exchange(other.features, 0))
    {
    }

    DeviceColumnarBuffer& operator=(DeviceColumnarBuffer&& other) noexcept
    {
        data = std::move(other.data);
        rows = std::exchange(other.rows, 0);
        features = std::exchange(other.features, 0);
        return *this;
    }

    [[nodiscard]] DeviceView view() const noexcept
    {
        return {data.get(), rows, features};
    }

    DeviceArray<float> data;
    std::size_t rows{0};
    std::size_t features{0};
};

void require_device_accessible(const void* pointer);
void copy_to_device(const float* host, DeviceColumnarBuffer& device);
void copy_to_host(DeviceView device, float* host);
void copy_device(const float* source, std::size_t count, float* destination);

}
