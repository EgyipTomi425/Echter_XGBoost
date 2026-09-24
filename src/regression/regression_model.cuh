#pragma once

#include "../core/cuda_utils.cuh"

#include <cstdint>
#include <string>
#include <vector>

namespace echter::xgb::detail
{

inline constexpr std::uint32_t default_left_flag = 1u << 31;

struct alignas(16) Node
{
    int left;
    int right;
    std::uint32_t split_feature;
    float value;
};

// Trees are stored back to back in `nodes`; `entry_nodes[tree]` is the index of
// each tree's root, and child indexes are absolute indexes into `nodes`.
struct HostModel
{
    std::vector<Node> nodes;
    std::vector<int> entry_nodes;
    int num_features{0};
    float base_score{0.0f};
};

struct DeviceModel
{
    DeviceArray<Node> nodes;
    DeviceArray<int> entry_nodes;
    int num_trees{0};
    int num_features{0};
    float base_score{0.0f};
};

// Parses and validates an XGBoost JSON regression model. Throws on failure.
HostModel load_model_with_cudf(const std::string& json_path);
DeviceModel upload_model(const HostModel& host);

}
