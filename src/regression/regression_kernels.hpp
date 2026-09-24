#pragma once

#include "regression_model.hpp"

#include <cstddef>

namespace echter::xgb::detail
{

void launch_regression_kernel(
    const float* device_features,
    float* device_output,
    std::size_t rows,
    const DeviceModel& model);

}
