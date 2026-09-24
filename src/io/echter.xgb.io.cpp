module;

#include "io_backend.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

module echter.xgb.io;

namespace echter::xgb::io
{

std::string read_json(const std::string& path)
{
    return io_backend::read_file(path);
}

DeviceColumnarData read_csv(const std::string& path, bool has_header)
{
    std::size_t rows = 0;
    std::size_t columns = 0;
    void* device = io_backend::read_csv(path, has_header, rows, columns);
    if (device == nullptr)
    {
        throw std::runtime_error("failed to read CSV to GPU: " + path);
    }

    return DeviceColumnarData::from_backend(device);
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
    if (excluded_columns.size() > source.features)
    {
        throw std::invalid_argument("too many excluded CSV columns");
    }

    void* device = model_backend::allocate_device(
        source.rows,
        source.features - excluded_columns.size());
    if (device == nullptr
        || !io_backend::select_columns(
            {source.data, source.rows, source.features},
            excluded_columns,
            device))
    {
        model_backend::destroy_device(device);
        throw std::runtime_error("failed to select device CSV columns");
    }
    return DeviceColumnarData::from_backend(device);
}

void write_csv(
    const std::string& path,
    const DeviceColumnarData& table,
    const std::vector<std::string>& names)
{
    const auto view = table.view();
    if (!io_backend::write_csv(
            path,
            {model_backend::DeviceView{view.data, view.rows, view.features}},
            names))
    {
        throw std::runtime_error("failed to write CSV: " + path);
    }
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

    if (!io_backend::write_csv(path, backend_columns, names))
    {
        throw std::runtime_error("failed to write CSV: " + path);
    }
}

}
