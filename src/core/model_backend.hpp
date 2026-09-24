#pragma once

#include <cstddef>

namespace echter::xgb::model_backend
{

struct DeviceView
{
    const float* data{nullptr};
    std::size_t rows{0};
    std::size_t features{0};
};

void* allocate_device(std::size_t rows, std::size_t features);
void destroy_device(void* device) noexcept;
DeviceView view(const void* device) noexcept;
bool upload(const float* host, std::size_t rows, std::size_t features, void* device);
bool download(DeviceView device, float* host, std::size_t elements);
bool select_columns(DeviceView source, std::size_t excluded, void* destination);
}
