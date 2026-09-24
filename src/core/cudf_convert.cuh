#pragma once

#include "device_buffer.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace echter::xgb::detail
{

inline void copy_column_as_float(
    const cudf::column_view& column,
    std::size_t index,
    float* destination)
{
    if (!cudf::is_numeric(column.type()))
    {
        throw std::invalid_argument(
            "column " + std::to_string(index) + " is not numeric");
    }

    cudf::column_view values = column;
    std::unique_ptr<cudf::column> converted;
    if (values.type().id() != cudf::type_id::FLOAT32)
    {
        converted = cudf::cast(values, cudf::data_type{cudf::type_id::FLOAT32});
        values = converted->view();
    }
    if (values.has_nulls())
    {
        converted = cudf::replace_nulls(
            values,
            cudf::numeric_scalar<float>(std::numeric_limits<float>::quiet_NaN()));
        values = converted->view();
    }

    copy_device(values.data<float>(), static_cast<std::size_t>(values.size()), destination);
}

}
