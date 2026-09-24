#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../core/model_backend.hpp"

namespace echter::xgb::io_backend
{

// Device tables are returned as model_backend handles. Failures throw.
std::string read_file(const std::string& path);
void* read_csv(const std::string& path, bool has_header);
void write_csv(
    const std::string& path,
    const std::vector<model_backend::DeviceView>& columns,
    const std::vector<std::string>& names);
void* select_columns(
    model_backend::DeviceView source,
    const std::vector<std::size_t>& excluded_columns);

}
