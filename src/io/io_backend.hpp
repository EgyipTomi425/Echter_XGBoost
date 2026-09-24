#pragma once

#include "../core/device_buffer.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace echter::xgb::io_backend
{

std::string read_file(
    const std::string& path,
    std::string_view prefix = {},
    std::string_view suffix = {});
detail::DeviceColumnarBuffer read_csv(const std::string& path, bool has_header);
void write_csv(
    const std::string& path,
    const std::vector<detail::DeviceView>& columns,
    const std::vector<std::string>& names);
detail::DeviceColumnarBuffer select_columns(
    detail::DeviceView source,
    const std::vector<std::size_t>& excluded_columns);

}
