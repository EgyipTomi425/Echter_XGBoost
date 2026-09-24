module;

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

export module echter.xgb.io;

export import echter.xgb.data;

export namespace echter::xgb::io
{

[[nodiscard]] std::string read_json(const std::string& path);
[[nodiscard]] DeviceColumnarData read_csv(
    const std::string& path,
    bool has_header = true);
[[nodiscard]] DeviceColumnarData select_columns(
    const DeviceColumnarData& table,
    std::size_t excluded_column);
[[nodiscard]] DeviceColumnarData select_columns(
    const DeviceColumnarData& table,
    const std::vector<std::size_t>& excluded_columns);
void write_csv(
    const std::string& path,
    const DeviceColumnarData& table,
    const std::vector<std::string>& names = {});
void write_csv(
    const std::string& path,
    const std::vector<DeviceColumnarView>& columns,
    const std::vector<std::string>& names = {});

}
