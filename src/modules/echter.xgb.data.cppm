module;

#include <cstddef>
#include <memory>
#include <vector>

export module echter.xgb.data;

export namespace echter::xgb
{

struct HostColumnarView
{
    const float* data{nullptr};
    std::size_t rows{0};
    std::size_t features{0};
};

struct DeviceColumnarView
{
    const float* data{nullptr};
    std::size_t rows{0};
    std::size_t features{0};
};

class DeviceColumnarData
{
public:
    DeviceColumnarData();
    ~DeviceColumnarData();

    DeviceColumnarData(DeviceColumnarData&&) noexcept;
    DeviceColumnarData& operator=(DeviceColumnarData&&) noexcept;

    DeviceColumnarData(const DeviceColumnarData&) = delete;
    DeviceColumnarData& operator=(const DeviceColumnarData&) = delete;

    [[nodiscard]] static DeviceColumnarData from_backend(void* device);

    [[nodiscard]] DeviceColumnarView view() const noexcept;
    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t features() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit DeviceColumnarData(std::unique_ptr<Impl> impl);
};

struct Prediction
{
    std::vector<float> values;
};

struct DevicePrediction
{
    DeviceColumnarData values;
};

}
