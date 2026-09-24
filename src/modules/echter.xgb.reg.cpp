module;

#include "../core/model_backend.hpp"
#include "../regression/regression_backend.hpp"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
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
}

Regression::~Regression() = default;
Regression::Regression(Regression&&) noexcept = default;
Regression& Regression::operator=(Regression&&) noexcept = default;

void Regression::load_model(const std::string& json_path)
{
    try
    {
        regression_backend::load_model(impl_->model, json_path);
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            "failed to load XGBoost regression model " + json_path + ": " + error.what());
    }

    impl_->loaded = true;
}

DeviceColumnarData Regression::upload(const HostColumnarView& host) const
{
    if (host.rows != 0 && host.features != 0 && host.data == nullptr)
    {
        throw std::invalid_argument("host input data is null");
    }

    void* device = model_backend::allocate_device(host.rows, host.features);
    auto result = DeviceColumnarData::from_backend(device);
    model_backend::upload(host.data, device);
    return result;
}

DeviceColumnarData Regression::allocate_device(
    std::size_t rows,
    std::size_t features) const
{
    return DeviceColumnarData::from_backend(
        model_backend::allocate_device(rows, features));
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

    if (device.features != num_features())
    {
        throw std::invalid_argument(
            "device input feature count does not match model: model expects "
            + std::to_string(num_features())
            + ", input has " + std::to_string(device.features));
    }

    void* output = model_backend::allocate_device(device.rows, 1);
    auto result = DeviceColumnarData::from_backend(output);
    regression_backend::predict_device(
        {device.data, device.rows, device.features},
        impl_->model,
        output);
    return {std::move(result)};
}

DevicePrediction Regression::predict(const DeviceColumnarData& device) const
{
    return predict(device.view());
}

Prediction Regression::predict(const HostColumnarView& host) const
{
    const auto device_prediction = predict(upload(host));
    const auto view = device_prediction.values.view();
    Prediction result;
    result.values.resize(host.rows);
    model_backend::download(
        {view.data, view.rows, view.features},
        result.values.data());
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
