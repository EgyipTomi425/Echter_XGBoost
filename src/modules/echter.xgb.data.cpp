module;

#include "../core/model_backend.hpp"

#include <memory>
#include <utility>

module echter.xgb.data;

namespace echter::xgb
{

struct DeviceColumnarData::Impl
{
    void* device{nullptr};

    ~Impl()
    {
        model_backend::destroy_device(device);
    }
};

DeviceColumnarData::DeviceColumnarData() = default;

DeviceColumnarData::~DeviceColumnarData() = default;

DeviceColumnarData::DeviceColumnarData(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

DeviceColumnarData DeviceColumnarData::from_backend(void* device)
{
    auto impl = std::make_unique<Impl>();
    impl->device = device;
    return DeviceColumnarData(std::move(impl));
}

DeviceColumnarData::DeviceColumnarData(DeviceColumnarData&&) noexcept = default;
DeviceColumnarData& DeviceColumnarData::operator=(DeviceColumnarData&&) noexcept = default;

DeviceColumnarView DeviceColumnarData::view() const noexcept
{
    if (!impl_)
    {
        return {};
    }

    const auto device = model_backend::view(impl_->device);
    return {device.data, device.rows, device.features};
}

std::size_t DeviceColumnarData::rows() const noexcept
{
    return view().rows;
}

std::size_t DeviceColumnarData::features() const noexcept
{
    return view().features;
}

bool DeviceColumnarData::empty() const noexcept
{
    return rows() == 0 || features() == 0;
}

}
