#pragma once

#include "../core/model_backend.hpp"

#include <string>

namespace echter::xgb::regression_backend
{

// Models are opaque handles, like device buffers. Failures throw.
void* create_model();
void destroy_model(void* model) noexcept;
void load_model(void* model, const std::string& path);
int model_features(const void* model) noexcept;
int model_trees(const void* model) noexcept;
void predict_device(
    model_backend::DeviceView device,
    const void* model,
    void* output);

}
