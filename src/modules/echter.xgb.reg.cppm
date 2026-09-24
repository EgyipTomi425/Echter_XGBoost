module;

#include "../regression/regression_model.hpp"

#include <cstddef>
#include <string>

export module echter.xgb.reg;

export import echter.xgb.data;

export namespace echter::xgb
{

class Regression
{
public:
    void load_model(const std::string& json_path);

    [[nodiscard]] DeviceColumnarData upload(const HostColumnarView& host) const;
    [[nodiscard]] DeviceColumnarData allocate_device(std::size_t rows, std::size_t features) const;

    [[nodiscard]] DevicePrediction predict(const DeviceColumnarView& device) const;
    [[nodiscard]] DevicePrediction predict(const DeviceColumnarData& device) const;
    [[nodiscard]] Prediction predict(const HostColumnarView& host) const;

    [[nodiscard]] std::size_t num_features() const noexcept;
    [[nodiscard]] std::size_t num_trees() const noexcept;

private:
    [[nodiscard]] bool loaded() const noexcept;

    detail::DeviceModel model_;
};

}
