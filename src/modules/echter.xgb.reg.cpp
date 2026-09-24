module;

#include "../core/model_backend.hpp"
#include "../regression/regression_backend.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

module echter.xgb.reg;

namespace echter::xgb
{

struct Regression::Impl
{
    void* model{nullptr};
    bool loaded{false};

    ~Impl()
    {
        regression_backend::destroy_model(model);
    }
};

Regression::Regression()
    : impl_(std::make_unique<Impl>())
{
    impl_->model = regression_backend::create_model();
    if (impl_->model == nullptr)
    {
        throw std::runtime_error("failed to create regression model");
    }
}

Regression::~Regression() = default;
Regression::Regression(Regression&&) noexcept = default;
Regression& Regression::operator=(Regression&&) noexcept = default;

void Regression::load_model(const std::string& json_path)
{
    if (!regression_backend::load_model(impl_->model, json_path))
    {
        throw std::runtime_error("failed to load XGBoost regression model: " + json_path);
    }

    impl_->loaded = true;
}

DeviceColumnarData Regression::upload(const HostColumnarView& host) const
{
    if (host.rows != 0 && host.features != 0 && host.data == nullptr)
    {
        throw std::invalid_argument("host input data is null");
    }

    if (host.rows != 0 && host.features > static_cast<std::size_t>(-1) / host.rows)
    {
        throw std::invalid_argument("host input size overflows");
    }

    void* device = model_backend::allocate_device(host.rows, host.features);
    if (device == nullptr
        || !model_backend::upload(host.data, host.rows, host.features, device))
    {
        model_backend::destroy_device(device);
        throw std::runtime_error("failed to upload columnar input to GPU");
    }

    return DeviceColumnarData::from_backend(device);
}

DeviceColumnarData Regression::allocate_device(
    std::size_t rows,
    std::size_t features) const
{
    void* device = model_backend::allocate_device(rows, features);
    if (device == nullptr)
    {
        throw std::runtime_error("failed to allocate columnar input on GPU");
    }

    return DeviceColumnarData::from_backend(device);
}

DevicePrediction Regression::predict(const DeviceColumnarView& device) const
{
    if (!impl_->loaded)
    {
        throw std::runtime_error("regression model is not loaded");
    }

    if (device.rows == 0)
    {
        return {};
    }

    if (device.data == nullptr)
    {
        throw std::invalid_argument("device input data is null");
    }

    if (device.features != static_cast<std::size_t>(regression_backend::model_features(impl_->model)))
    {
        throw std::invalid_argument(
            "device input feature count does not match model: model expects "
            + std::to_string(regression_backend::model_features(impl_->model))
            + ", input has " + std::to_string(device.features));
    }

    void* output = model_backend::allocate_device(device.rows, 1);
    if (output == nullptr
        || !regression_backend::predict_device(
            {device.data, device.rows, device.features},
            impl_->model,
            output))
    {
        model_backend::destroy_device(output);
        throw std::runtime_error("GPU regression inference failed");
    }

    return {DeviceColumnarData::from_backend(output)};
}

DevicePrediction Regression::predict(const DeviceColumnarData& device) const
{
    return predict(device.view());
}

Prediction Regression::predict(const HostColumnarView& host) const
{
    const auto device = upload(host);
    const auto device_prediction = predict(device.view());
    Prediction result;
    result.values.resize(host.rows);
    if (!model_backend::download(
            {device_prediction.values.view().data,
             device_prediction.values.view().rows,
             device_prediction.values.view().features},
            result.values.data(),
            host.rows))
    {
        throw std::runtime_error("failed to copy regression predictions to host");
    }
    return result;
}

std::size_t Regression::num_features() const noexcept
{
    return impl_->loaded
        ? static_cast<std::size_t>(regression_backend::model_features(impl_->model))
        : 0;
}

std::size_t Regression::num_trees() const noexcept
{
    return impl_->loaded
        ? static_cast<std::size_t>(regression_backend::model_trees(impl_->model))
        : 0;
}

}
