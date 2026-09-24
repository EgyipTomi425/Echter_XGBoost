#include "../core/cudf_convert.cuh"

#include <cudf/table/table_view.hpp>

#include <stdexcept>
#include <string>

import echter.xgb.reg;

#include "regression_cudf_adapter.hpp"

namespace echter::xgb
{

DevicePrediction predict(
    const Regression& regression,
    const cudf::table_view& table)
{
    const auto rows = static_cast<std::size_t>(table.num_rows());
    const auto features = static_cast<std::size_t>(table.num_columns());

    if (features != regression.num_features())
    {
        throw std::invalid_argument(
            "cuDF table feature count does not match model: model expects "
            + std::to_string(regression.num_features())
            + ", input has " + std::to_string(features));
    }

    if (rows == 0)
    {
        return {};
    }

    // cuDF stores each column separately. The public regression API accepts
    // one primitive contiguous device buffer, so this adapter packs the
    // columns into the library-owned column-major GPU buffer without a
    // device-to-host round trip.
    auto packed = regression.allocate_device(rows, features);
    auto* const destination = const_cast<float*>(packed.view().data);
    for (std::size_t feature = 0; feature < features; ++feature)
    {
        detail::copy_column_as_float(
            table.column(static_cast<cudf::size_type>(feature)),
            feature,
            destination + feature * rows);
    }

    return regression.predict(packed);
}

}
