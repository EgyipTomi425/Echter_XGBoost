#include "io_backend.hpp"

#include "../core/cuda_utils.cuh"
#include "../core/cudf_convert.cuh"

#include <cuda_runtime.h>

#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/table/table_view.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace echter::xgb::io_backend
{

namespace
{

// cuDF reports missing files with a long low-level message.
void require_file(const std::string& path)
{
    if (!std::filesystem::is_regular_file(path))
    {
        throw std::runtime_error("file not found");
    }
}

}

std::string read_file(const std::string& path)
{
    require_file(path);
    const auto source = cudf::io::datasource::create(path);
    std::string contents(source->size(), '\0');
    const auto bytes = source->host_read(
        0,
        source->size(),
        reinterpret_cast<std::uint8_t*>(contents.data()));
    contents.resize(bytes);
    return contents;
}

void* read_csv(const std::string& path, bool has_header)
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
    auto buffer = std::make_unique<detail::DeviceColumnarBuffer>(rows, columns);
    for (std::size_t column = 0; column < columns; ++column)
    {
        detail::copy_column_as_float(
            table.column(static_cast<cudf::size_type>(column)),
            column,
            buffer->data.get() + column * rows);
    }
    return buffer.release();
}

void write_csv(
    const std::string& path,
    const std::vector<model_backend::DeviceView>& columns,
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

    // The table views the caller's device memory directly, so nothing is copied
    // before cuDF formats the output.
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

void* select_columns(
    model_backend::DeviceView source,
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

    auto target = std::make_unique<detail::DeviceColumnarBuffer>(
        source.rows,
        source.features - excluded_columns.size());
    if (source.rows == 0)
    {
        return target.release();
    }

    std::size_t target_column = 0;
    for (std::size_t source_column = 0; source_column < source.features; ++source_column)
    {
        if (excluded[source_column])
        {
            continue;
        }

        detail::check_cuda(
            cudaMemcpy(
                target->data.get() + target_column * source.rows,
                source.data + source_column * source.rows,
                source.rows * sizeof(float),
                cudaMemcpyDeviceToDevice),
            "cudaMemcpy(select columns D2D)");
        ++target_column;
    }
    return target.release();
}

}
