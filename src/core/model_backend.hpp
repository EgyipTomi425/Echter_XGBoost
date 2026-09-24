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

// Device buffers are passed around as opaque handles so that module units stay
// free of CUDA headers. All functions report failures with exceptions.
void* allocate_device(std::size_t rows, std::size_t features);
void destroy_device(void* device) noexcept;
DeviceView view(const void* device) noexcept;
void upload(const float* host, void* device);
void download(DeviceView device, float* host);

}
