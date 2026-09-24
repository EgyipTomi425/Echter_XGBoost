#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../core/model_backend.hpp"

namespace echter::xgb::io_backend
{

std::string read_file(const std::string& path);
void* read_csv(
    const std::string& path,
    bool has_header,
    std::size_t& rows,
    std::size_t& columns);
bool write_csv(
    const std::string& path,
    const std::vector<model_backend::DeviceView>& columns,
    const std::vector<std::string>& names);
bool select_columns(
    model_backend::DeviceView source,
    const std::vector<std::size_t>& excluded_columns,
    void* destination);

}
