#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace echter::xgb::detail
{

inline void check_cuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct CudaFree
{
    void operator()(void* pointer) const noexcept
    {
        cudaFree(pointer);
    }
};

template <typename T>
using DeviceArray = std::unique_ptr<T[], CudaFree>;

template <typename T>
DeviceArray<T> allocate_device_array(std::size_t count)
{
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
        throw std::length_error("device allocation size overflows");
    }

    T* pointer = nullptr;
    if (count != 0)
    {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)),
            "cudaMalloc");
    }
    return DeviceArray<T>(pointer);
}

// Column-major float matrix in device memory: data[feature * rows + row].
struct DeviceColumnarBuffer
{
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

    DeviceArray<float> data;
    std::size_t rows{0};
    std::size_t features{0};
};

}
