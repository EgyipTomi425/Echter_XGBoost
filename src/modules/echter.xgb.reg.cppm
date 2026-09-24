module;

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
export module echter.xgb.reg;

export import echter.xgb.data;

export namespace echter::xgb
{

using RegressionPrediction = Prediction;

class Regression
{
public:
    Regression();
    ~Regression();

    Regression(Regression&&) noexcept;
    Regression& operator=(Regression&&) noexcept;

    Regression(const Regression&) = delete;
    Regression& operator=(const Regression&) = delete;

    void load_model(const std::string& json_path);

    [[nodiscard]] DeviceColumnarData upload(const HostColumnarView& host) const;
    [[nodiscard]] DeviceColumnarData allocate_device(std::size_t rows, std::size_t features) const;

    [[nodiscard]] DevicePrediction predict(const DeviceColumnarView& device) const;
    [[nodiscard]] DevicePrediction predict(const DeviceColumnarData& device) const;
    [[nodiscard]] Prediction predict(const HostColumnarView& host) const;

    [[nodiscard]] std::size_t num_features() const noexcept;
    [[nodiscard]] std::size_t num_trees() const noexcept;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}
