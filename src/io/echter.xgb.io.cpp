module;

#include "io_backend.hpp"

#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

module echter.xgb.io;

namespace echter::xgb::io
{

std::string read_json(const std::string& path)
{
    try
    {
        return io_backend::read_file(path);
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error("failed to read JSON " + path + ": " + error.what());
    }
}

DeviceColumnarData read_csv(const std::string& path, bool has_header)
{
    try
    {
        return DeviceColumnarData::from_backend(io_backend::read_csv(path, has_header));
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error("failed to read CSV " + path + ": " + error.what());
    }
}

DeviceColumnarData select_columns(
    const DeviceColumnarData& table,
    std::size_t excluded_column)
{
    return select_columns(table, std::vector<std::size_t>{excluded_column});
}

DeviceColumnarData select_columns(
    const DeviceColumnarData& table,
    const std::vector<std::size_t>& excluded_columns)
{
    const auto source = table.view();
    return DeviceColumnarData::from_backend(io_backend::select_columns(
        {source.data, source.rows, source.features},
        excluded_columns));
}

void write_csv(
    const std::string& path,
    const DeviceColumnarData& table,
    const std::vector<std::string>& names)
{
    const auto view = table.view();
    std::vector<DeviceColumnarView> columns;
    columns.reserve(view.features);
    for (std::size_t feature = 0; feature < view.features; ++feature)
    {
        columns.push_back({view.data + feature * view.rows, view.rows, 1});
    }
    write_csv(path, columns, names);
}

void write_csv(
    const std::string& path,
    const std::vector<DeviceColumnarView>& columns,
    const std::vector<std::string>& names)
{
    std::vector<model_backend::DeviceView> backend_columns;
    backend_columns.reserve(columns.size());
    for (const auto& column : columns)
    {
        backend_columns.push_back({column.data, column.rows, column.features});
    }

    try
    {
        io_backend::write_csv(path, backend_columns, names);
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error("failed to write CSV " + path + ": " + error.what());
    }
}

}
