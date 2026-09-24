#include "device_buffer.hpp"

#include "cuda_check.hpp"

#include <cuda_runtime.h>

namespace echter::xgb::detail
{

namespace
{

bool pageable_memory_accessible()
{
    int device = 0;
    int accessible = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(
        cudaDeviceGetAttribute(&accessible, cudaDevAttrPageableMemoryAccess, device),
        "cudaDeviceGetAttribute");
    return accessible != 0;
}

}

void DeviceFree::operator()(void* pointer) const noexcept
{
    cudaFree(pointer);
}

void* allocate_device_bytes(std::size_t bytes)
{
    void* pointer = nullptr;
    if (bytes != 0)
    {
        check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
    }
    return pointer;
}

void require_device_accessible(const void* pointer)
{
    cudaPointerAttributes attributes{};
    check_cuda(cudaPointerGetAttributes(&attributes, pointer), "cudaPointerGetAttributes");
    if (attributes.type == cudaMemoryTypeUnregistered && !pageable_memory_accessible())
    {
        throw std::invalid_argument("device data points to host memory");
    }
}

void copy_to_device(const float* host, DeviceColumnarBuffer& device)
{
    const std::size_t elements = device.rows * device.features;
    if (elements != 0)
    {
        check_cuda(
            cudaMemcpy(device.data.get(), host, elements * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy(host to device)");
    }
}

void copy_to_host(DeviceView device, float* host)
{
    const std::size_t elements = device.rows * device.features;
    if (elements != 0)
    {
        check_cuda(
            cudaMemcpy(host, device.data, elements * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy(device to host)");
    }
}

void copy_device(const float* source, std::size_t count, float* destination)
{
    if (count != 0)
    {
        check_cuda(
            cudaMemcpy(destination, source, count * sizeof(float), cudaMemcpyDeviceToDevice),
            "cudaMemcpy(device to device)");
    }
}

}
