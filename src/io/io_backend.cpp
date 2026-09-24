#include "io_backend.hpp"

#include "../core/cudf_convert.cuh"

#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/table/table_view.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace echter::xgb::io_backend
{

namespace
{

void require_file(const std::string& path)
{
    if (!std::filesystem::is_regular_file(path))
    {
        throw std::runtime_error("file not found");
    }
}

}

std::string read_file(
    const std::string& path,
    std::string_view prefix,
    std::string_view suffix)
{
    require_file(path);
    const auto source = cudf::io::datasource::create(path);
    const std::size_t size = source->size();
    std::string contents(prefix.size() + size + suffix.size(), '\0');
    const std::size_t bytes = source->host_read(
        0,
        size,
        reinterpret_cast<std::uint8_t*>(contents.data() + prefix.size()));
    contents.resize(prefix.size() + bytes + suffix.size());
    prefix.copy(contents.data(), prefix.size());
    suffix.copy(contents.data() + prefix.size() + bytes, suffix.size());
    return contents;
}

detail::DeviceColumnarBuffer read_csv(const std::string& path, bool has_header)
{
    require_file(path);
    const auto options = cudf::io::csv_reader_options::builder(
        cudf::io::source_info(path))
        .header(has_header ? 0 : -1)
        .build();
    const auto input = cudf::io::read_csv(options);
    const auto table = input.tbl->view();

    const auto rows = static_cast<std::size_t>(table.num_rows());
    const auto columns = static_cast<std::size_t>(table.num_columns());
    detail::DeviceColumnarBuffer buffer(rows, columns);
    for (std::size_t column = 0; column < columns; ++column)
    {
        detail::copy_column_as_float(
            table.column(static_cast<cudf::size_type>(column)),
            column,
            buffer.data.get() + column * rows);
    }
    return buffer;
}

void write_csv(
    const std::string& path,
    const std::vector<detail::DeviceView>& columns,
    const std::vector<std::string>& names)
{
    if (columns.empty())
    {
        throw std::invalid_argument("no columns to write");
    }
    if (!names.empty() && names.size() != columns.size())
    {
        throw std::invalid_argument(
            "CSV header has " + std::to_string(names.size()) + " names for "
            + std::to_string(columns.size()) + " columns");
    }

    const auto rows = columns.front().rows;
    if (rows > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()))
    {
        throw std::length_error("too many rows for a cuDF table");
    }

    std::vector<cudf::column_view> views;
    views.reserve(columns.size());
    for (const auto& column : columns)
    {
        if (column.rows != rows || column.features != 1
            || (rows != 0 && column.data == nullptr))
        {
            throw std::invalid_argument(
                "CSV output columns must be single, non-null columns of equal length");
        }
        if (rows != 0)
        {
            detail::require_device_accessible(column.data);
        }
        views.emplace_back(
            cudf::data_type{cudf::type_id::FLOAT32},
            static_cast<cudf::size_type>(rows),
            column.data,
            nullptr,
            0);
    }

    auto builder = cudf::io::csv_writer_options::builder(
        cudf::io::sink_info(path),
        cudf::table_view(views))
        .include_header(!names.empty());
    if (!names.empty())
    {
        builder.names(names);
    }
    cudf::io::write_csv(builder.build());
}

detail::DeviceColumnarBuffer select_columns(
    detail::DeviceView source,
    const std::vector<std::size_t>& excluded_columns)
{
    std::vector<bool> excluded(source.features, false);
    for (const auto column : excluded_columns)
    {
        if (column >= source.features)
        {
            throw std::out_of_range(
                "excluded column " + std::to_string(column) + " is out of range for "
                + std::to_string(source.features) + " columns");
        }
        if (excluded[column])
        {
            throw std::invalid_argument(
                "column " + std::to_string(column) + " is excluded more than once");
        }
        excluded[column] = true;
    }

    detail::DeviceColumnarBuffer target(
        source.rows,
        source.features - excluded_columns.size());
    std::size_t target_column = 0;
    for (std::size_t source_column = 0; source_column < source.features; ++source_column)
    {
        if (!excluded[source_column])
        {
            detail::copy_device(
                source.data + source_column * source.rows,
                source.rows,
                target.data.get() + target_column++ * source.rows);
        }
    }
    return target;
}

}
