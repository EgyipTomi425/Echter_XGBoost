#pragma once

#include <cstddef>

#include "regression_model.cuh"

namespace echter::xgb::detail
{

// Launches the prediction kernel asynchronously. The caller checks for errors.
void launch_regression_kernel(
    const float* device_features,
    float* device_output,
    std::size_t rows,
    const DeviceModel& model);

}
