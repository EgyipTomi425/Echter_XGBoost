#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace echter::xgb::detail
{

struct Node
{
    int feature;
    float threshold;
    int left;
    int right;
    float leaf;
    std::int8_t is_leaf;
    std::int8_t default_left;
};

struct HostModel
{
    std::vector<Node> nodes;
    std::vector<int> entry_nodes;
    std::vector<std::string> feature_names;
    int num_features{0};
    float base_score{0.0f};
};

struct DeviceModel
{
    DeviceModel() = default;
    ~DeviceModel();

    DeviceModel(const DeviceModel&) = delete;
    DeviceModel& operator=(const DeviceModel&) = delete;

    DeviceModel(DeviceModel&& other) noexcept;
    DeviceModel& operator=(DeviceModel&& other) noexcept;

    Node* nodes{nullptr};
    int* entry_nodes{nullptr};
    int num_nodes{0};
    int num_trees{0};
    int num_features{0};
    float base_score{0.0f};

    void release() noexcept;
};

bool load_model_with_cudf(const std::string& json_path, HostModel& out);
bool upload_model(const HostModel& host, DeviceModel& device);

}
