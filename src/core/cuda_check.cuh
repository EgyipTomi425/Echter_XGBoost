#pragma once

#include <cuda_runtime.h>

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

}
