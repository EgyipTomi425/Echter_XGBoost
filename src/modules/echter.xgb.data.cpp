module;

#include "../core/device_buffer.hpp"

#include <cstddef>
#include <utility>

module echter.xgb.data;

namespace echter::xgb
{

DeviceColumnarData::DeviceColumnarData(detail::DeviceColumnarBuffer buffer) noexcept
    : buffer_(std::move(buffer))
{
}

DeviceColumnarView DeviceColumnarData::view() const noexcept
{
    return {buffer_.data.get(), buffer_.rows, buffer_.features};
}

float* DeviceColumnarData::mutable_data() noexcept
{
    return buffer_.data.get();
}

std::size_t DeviceColumnarData::rows() const noexcept
{
    return buffer_.rows;
}

std::size_t DeviceColumnarData::features() const noexcept
{
    return buffer_.features;
}

bool DeviceColumnarData::empty() const noexcept
{
    return rows() == 0 || features() == 0;
}

}
