module;

#include "../core/device_buffer.hpp"

#include <cstddef>
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
    DeviceColumnarData() = default;
    explicit DeviceColumnarData(detail::DeviceColumnarBuffer buffer) noexcept;

    [[nodiscard]] DeviceColumnarView view() const noexcept;
    [[nodiscard]] float* mutable_data() noexcept;
    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t features() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

private:
    detail::DeviceColumnarBuffer buffer_;
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
