#pragma once

#include "../core/device_buffer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace echter::xgb::detail
{

struct alignas(8) Node
{
    std::uint32_t link;
    float value;
};

struct HostModel
{
    std::vector<Node> nodes;
    std::vector<int> entry_nodes;
    int num_features{0};
    float base_score{0.0f};
    std::uint32_t feature_bits{0};
};

struct DeviceModel
{
    DeviceArray<Node> nodes;
    DeviceArray<int> entry_nodes;
    int num_trees{0};
    int num_features{0};
    float base_score{0.0f};
    std::uint32_t feature_bits{0};
};

HostModel parse_model_json(const std::string& json_path);
DeviceModel upload_model(const HostModel& host);
DeviceColumnarBuffer predict(const DeviceModel& model, DeviceView input);

}
