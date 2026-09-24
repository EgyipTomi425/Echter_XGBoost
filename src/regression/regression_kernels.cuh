#pragma once

#include <cstddef>

#include "regression_model.cuh"

namespace echter::xgb::detail
{

void launch_regression_kernel(
    const float* device_features,
    float* device_output,
    std::size_t rows,
    std::size_t features,
    const DeviceModel& model);

}
