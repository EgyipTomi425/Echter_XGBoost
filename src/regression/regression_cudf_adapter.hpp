#pragma once

namespace cudf { class table_view; }

namespace echter::xgb
{

[[nodiscard]] DevicePrediction predict(
    const Regression& regression,
    const cudf::table_view& table);

}
