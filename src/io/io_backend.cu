#include "io_backend.hpp"

#include "../core/model_backend.hpp"

#include <cuda_runtime.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <memory>
#include <stdexcept>
#include <cstdint>
#include <vector>

namespace echter::xgb::io_backend
{

std::string read_file(const std::string& path)
{
    auto source = cudf::io::datasource::create(path);
    if (!source)
    {
        throw std::runtime_error("cannot open file for reading: " + path);
    }

    std::string contents(source->size(), '\0');
    const auto bytes = source->host_read(
        0,
        source->size(),
        reinterpret_cast<std::uint8_t*>(contents.data()));
    contents.resize(bytes);
    return contents;
}

void* read_csv(
    const std::string& path,
    bool has_header,
    std::size_t& rows,
    std::size_t& columns)
{
    const auto options = cudf::io::csv_reader_options::builder(
        cudf::io::source_info(path))
        .header(has_header ? 0 : -1)
        .build();
    auto input = cudf::io::read_csv(options);
    if (!input.tbl)
    {
        throw std::runtime_error("cuDF returned no table for CSV: " + path);
    }

    rows = static_cast<std::size_t>(input.tbl->num_rows());
    columns = static_cast<std::size_t>(input.tbl->num_columns());
    void* device = model_backend::allocate_device(rows, columns);
    if (device == nullptr)
    {
        throw std::runtime_error("failed to allocate CSV device table");
    }

    const auto destination = model_backend::view(device);
    try
    {
        for (std::size_t column = 0; column < columns; ++column)
        {
            auto source = input.tbl->view().column(
                static_cast<cudf::size_type>(column));
            std::unique_ptr<cudf::column> converted;
            if (source.type().id() != cudf::type_id::FLOAT32)
            {
                converted = cudf::cast(
                    source,
                    cudf::data_type{cudf::type_id::FLOAT32});
                source = converted->view();
            }

            if (cudaMemcpy(
                    const_cast<float*>(destination.data) + column * rows,
                    source.data<float>(),
                    rows * sizeof(float),
                    cudaMemcpyDeviceToDevice) != cudaSuccess)
            {
                throw std::runtime_error("failed to copy CSV column to GPU");
            }
        }
    }
    catch (...)
    {
        model_backend::destroy_device(device);
        throw;
    }

    return device;
}

bool write_csv(
    const std::string& path,
    const std::vector<model_backend::DeviceView>& columns,
    const std::vector<std::string>& names)
{
    if (columns.empty())
    {
        return false;
    }

    std::vector<std::unique_ptr<cudf::column>> output;
    output.reserve(columns.size());
    const auto rows = columns.front().rows;
    for (const auto& input : columns)
    {
        if (input.rows != rows || input.features != 1
            || (rows != 0 && input.data == nullptr))
        {
            return false;
        }
        auto result = cudf::make_numeric_column(
            cudf::data_type{cudf::type_id::FLOAT32},
            static_cast<cudf::size_type>(rows));
        if (rows != 0 && cudaMemcpy(
                result->mutable_view().data<float>(),
                input.data,
                rows * sizeof(float),
                cudaMemcpyDeviceToDevice) != cudaSuccess)
        {
            return false;
        }
        output.push_back(std::move(result));
    }

    cudf::table table(std::move(output));
    auto options = cudf::io::csv_writer_options::builder(
        cudf::io::sink_info(path), table.view())
        .include_header(!names.empty())
        .build();
    if (!names.empty() && names.size() == columns.size())
    {
        options = cudf::io::csv_writer_options::builder(
            cudf::io::sink_info(path), table.view())
            .include_header(true)
            .names(names)
            .build();
    }
    cudf::io::write_csv(options);
    return true;
}

bool select_columns(
    model_backend::DeviceView source,
    const std::vector<std::size_t>& excluded_columns,
    void* destination)
{
    if (destination == nullptr || excluded_columns.size() > source.features)
    {
        return false;
    }

    std::vector<bool> excluded(source.features, false);
    for (const auto column : excluded_columns)
    {
        if (column >= source.features || excluded[column])
        {
            return false;
        }
        excluded[column] = true;
    }

    const auto target = model_backend::view(destination);
    if (target.rows != source.rows
        || target.features != source.features - excluded_columns.size())
    {
        return false;
    }

    std::size_t target_column = 0;
    for (std::size_t source_column = 0; source_column < source.features; ++source_column)
    {
        if (excluded[source_column])
        {
            continue;
        }

        if (source.rows != 0 && (source.data == nullptr || target.data == nullptr))
        {
            return false;
        }
        if (cudaMemcpy(
                const_cast<float*>(target.data) + target_column * source.rows,
                source.data + source_column * source.rows,
                source.rows * sizeof(float),
                cudaMemcpyDeviceToDevice) != cudaSuccess)
        {
            return false;
        }
        ++target_column;
    }
    return true;
}

}
