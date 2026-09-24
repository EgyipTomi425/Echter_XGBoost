#include <cuda_runtime.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

    std::vector<std::unique_ptr<cudf::column>> casted;
    casted.reserve(features);
    std::vector<float*> columns(features, nullptr);

    for (std::size_t i = 0; i < features; ++i)
    {
        const auto column = table.column(static_cast<cudf::size_type>(i));

        if (column.type().id() == cudf::type_id::FLOAT32)
        {
            columns[i] = const_cast<float*>(column.data<float>());
        }
        else
        {
            auto converted = cudf::cast(
                column,
                cudf::data_type{cudf::type_id::FLOAT32});

            columns[i] = const_cast<float*>(converted->view().data<float>());
            casted.push_back(std::move(converted));
        }
    }

    // cuDF stores each column separately. The public regression API accepts
    // one primitive contiguous device buffer, so this adapter packs the
    // columns into the library-owned column-major GPU buffer without a
    // device-to-host round trip.
    auto packed = regression.allocate_device(rows, features);

    for (std::size_t feature = 0; feature < features; ++feature)
    {
        cudaError_t error = cudaMemcpy(
            const_cast<float*>(packed.view().data) + feature * rows,
            columns[feature],
            rows * sizeof(float),
            cudaMemcpyDeviceToDevice);

        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("cuDF column device copy failed: ")
                + cudaGetErrorString(error));
        }
    }

    return regression.predict(packed);
}

}
