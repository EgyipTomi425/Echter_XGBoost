module;

#include "../core/device_buffer.hpp"
#include "../regression/regression_model.hpp"

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

module echter.xgb.reg;

namespace echter::xgb
{

void Regression::load_model(const std::string& json_path)
{
    try
    {
        model_ = detail::upload_model(detail::parse_model_json(json_path));
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            "failed to load XGBoost regression model " + json_path + ": " + error.what());
    }
}

DeviceColumnarData Regression::upload(const HostColumnarView& host) const
{
    if (host.rows != 0 && host.features != 0 && host.data == nullptr)
    {
        throw std::invalid_argument("host input data is null");
    }

    detail::DeviceColumnarBuffer buffer(host.rows, host.features);
    detail::copy_to_device(host.data, buffer);
    return DeviceColumnarData(std::move(buffer));
}

DeviceColumnarData Regression::allocate_device(
    std::size_t rows,
    std::size_t features) const
{
    return DeviceColumnarData(detail::DeviceColumnarBuffer(rows, features));
}

DevicePrediction Regression::predict(const DeviceColumnarView& device) const
{
    if (!loaded())
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

    return {DeviceColumnarData(
        detail::predict(model_, {device.data, device.rows, device.features}))};
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
    detail::copy_to_host({view.data, view.rows, view.features}, result.values.data());
    return result;
}

std::size_t Regression::num_features() const noexcept
{
    return loaded() ? static_cast<std::size_t>(model_.num_features) : 0;
}

std::size_t Regression::num_trees() const noexcept
{
    return loaded() ? static_cast<std::size_t>(model_.num_trees) : 0;
}

bool Regression::loaded() const noexcept
{
    return model_.nodes != nullptr;
}

}
