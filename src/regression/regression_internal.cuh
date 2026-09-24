#pragma once

#include <cstddef>
#include <vector>

#include "regression_model.cuh"

namespace echter::xgb::detail
{

struct DeviceColumnarBuffer
{
    float* data{nullptr};
    std::size_t rows{0};
    std::size_t features{0};

    void release() noexcept;
};

bool allocate_columnar(
    std::size_t rows,
    std::size_t features,
    DeviceColumnarBuffer& out);

bool upload_columnar(
    const float* host,
    std::size_t rows,
    std::size_t features,
    DeviceColumnarBuffer& out);

bool run_regression_device(
    const DeviceColumnarBuffer& input,
    const DeviceModel& model,
    DeviceColumnarBuffer& output);

}
