#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

import echter.xgb;

#include "regression_cudf_adapter.hpp"

namespace
{

namespace xgb = echter::xgb;
using clock_type = std::chrono::steady_clock;

double median_ms(int repetitions, const std::function<void()>& run)
{
    run();
    std::vector<double> times;
    for (int repetition = 0; repetition < repetitions; ++repetition)
    {
        const auto start = clock_type::now();
        run();
        times.push_back(std::chrono::duration<double, std::milli>(clock_type::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

std::vector<float> to_host(const xgb::DeviceColumnarView& view)
{
    std::vector<float> host(view.rows * view.features);
    cudaMemcpy(host.data(), view.data, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return host;
}

void write_floats(const std::filesystem::path& path, const std::vector<float>& values)
{
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
}

}

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::cerr << "usage: echter_benchmark <model.json> <output-dir> <input.csv> "
                     "<input_x10.csv> <input_x20.csv>\n";
        return 2;
    }

    const std::filesystem::path output_dir = argv[2];
    const std::vector<std::pair<std::size_t, std::string>> batches{
        {1, argv[3]}, {10, argv[4]}, {20, argv[5]}};
    std::filesystem::create_directories(output_dir);

    xgb::Regression regression;
    regression.load_model(argv[1]);

    const auto base_table = xgb::io::select_columns(xgb::io::read_csv(batches.front().second), {0, 1});
    const auto base = to_host(base_table.view());
    const std::size_t base_rows = base_table.rows();
    const std::size_t features = base_table.features();
    {
        std::ofstream out(output_dir / "features.bin", std::ios::binary);
        const std::uint64_t shape[2]{base_rows, features};
        out.write(reinterpret_cast<const char*>(shape), sizeof(shape));
        out.write(reinterpret_cast<const char*>(base.data()), base.size() * sizeof(float));
    }

    std::cout << std::fixed << std::setprecision(2);
    for (const auto& [factor, csv] : batches)
    {
        const std::size_t rows = base_rows * factor;
        std::vector<float> host(rows * features);
        for (std::size_t feature = 0; feature < features; ++feature)
        {
            for (std::size_t copy = 0; copy < factor; ++copy)
            {
                std::copy_n(base.data() + feature * base_rows, base_rows,
                            host.data() + feature * rows + copy * base_rows);
            }
        }
        const xgb::HostColumnarView host_view{host.data(), rows, features};

        const double host_ms = median_ms(9, [&] { (void)regression.predict(host_view); });
        const auto host_prediction = regression.predict(host_view).values;

        const auto device = regression.upload(host_view);
        const double device_ms = median_ms(9, [&] { (void)regression.predict(device); });
        const auto device_prediction = to_host(regression.predict(device).values.view());

        std::vector<std::unique_ptr<cudf::column>> columns;
        for (std::size_t feature = 0; feature < features; ++feature)
        {
            auto column = cudf::make_numeric_column(
                cudf::data_type{cudf::type_id::FLOAT32}, static_cast<cudf::size_type>(rows));
            cudaMemcpy(column->mutable_view().data<float>(), device.view().data + feature * rows,
                       rows * sizeof(float), cudaMemcpyDeviceToDevice);
            columns.push_back(std::move(column));
        }
        const cudf::table table(std::move(columns));
        const double cudf_ms = median_ms(9, [&] { (void)xgb::predict(regression, table.view()); });
        const auto cudf_prediction = to_host(xgb::predict(regression, table.view()).values.view());

        const double csv_ms = median_ms(5, [&] {
            const auto input = xgb::io::read_csv(csv);
            (void)regression.predict(xgb::io::select_columns(input, {0, 1}));
        });
        const auto csv_input = xgb::io::read_csv(csv);
        const auto csv_prediction =
            to_host(regression.predict(xgb::io::select_columns(csv_input, {0, 1})).values.view());

        const bool paths_agree = device_prediction == host_prediction
            && cudf_prediction == host_prediction && csv_prediction == host_prediction;
        const auto rate = [rows](double ms) { return rows / ms / 1000.0; };
        std::cout << "rows " << std::setw(8) << rows
                  << " | device " << device_ms << " ms " << rate(device_ms) << " M/s"
                  << " | cuDF " << cudf_ms << " ms " << rate(cudf_ms) << " M/s"
                  << " | host " << host_ms << " ms " << rate(host_ms) << " M/s"
                  << " | CSV " << csv_ms << " ms " << rate(csv_ms) << " M/s"
                  << " | paths agree: " << (paths_agree ? "yes" : "no") << '\n';

        write_floats(output_dir / ("echter_x" + std::to_string(factor) + ".bin"), device_prediction);
    }
}
