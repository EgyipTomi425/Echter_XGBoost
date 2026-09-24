#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

import echter.xgb;
#include "../src/regression/regression_cudf_adapter.hpp"

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "usage: echter_xgb_cudf_prediction <model.json> [rows]\n";
        return 2;
    }

    try
    {
        const std::size_t rows = argc == 3 ? std::stoull(argv[2]) : 100000;
        echter::xgb::XGBoost api;
        auto& regression = api.regression();
        regression.load_model(argv[1]);

        const std::size_t features = regression.num_features();
        std::mt19937 generator(12345u);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<std::unique_ptr<cudf::column>> columns;
        columns.reserve(features);
        for (std::size_t feature = 0; feature < features; ++feature)
        {
            auto column = cudf::make_numeric_column(
                cudf::data_type{cudf::type_id::FLOAT32},
                static_cast<cudf::size_type>(rows));
            std::vector<float> host_values(rows);
            for (float& value : host_values)
            {
                value = distribution(generator);
            }
            if (rows != 0 && cudaMemcpy(
                    column->mutable_view().data<float>(),
                    host_values.data(),
                    rows * sizeof(float),
                    cudaMemcpyHostToDevice) != cudaSuccess)
            {
                throw std::runtime_error("failed to create cuDF example input");
            }
            columns.push_back(std::move(column));
        }

        cudf::table table(std::move(columns));
        const auto start = std::chrono::steady_clock::now();
        const auto prediction = echter::xgb::predict(regression, table.view());
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            throw std::runtime_error("cuDF prediction synchronization failed");
        }
        const auto end = std::chrono::steady_clock::now();

        std::vector<float> first_predictions(std::min<std::size_t>(10, rows));
        std::vector<float> all_predictions(rows);
        if (!all_predictions.empty()
            && cudaMemcpy(
                all_predictions.data(),
                prediction.values.view().data,
                rows * sizeof(float),
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            throw std::runtime_error("failed to read cuDF example statistics");
        }
        if (!first_predictions.empty()
            && cudaMemcpy(
                first_predictions.data(),
                prediction.values.view().data,
                first_predictions.size() * sizeof(float),
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            throw std::runtime_error("failed to read cuDF example output");
        }

        std::size_t finite_predictions = 0;
        double sum = 0.0;
        double squared_sum = 0.0;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        for (const float value : all_predictions)
        {
            if (!std::isfinite(value))
            {
                continue;
            }
            ++finite_predictions;
            sum += value;
            squared_sum += static_cast<double>(value) * value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }

        const double mean = finite_predictions == 0
            ? std::numeric_limits<double>::quiet_NaN()
            : sum / finite_predictions;
        const double variance = finite_predictions == 0
            ? std::numeric_limits<double>::quiet_NaN()
            : std::max(0.0, squared_sum / finite_predictions - mean * mean);

        std::cout << "cuDF prediction\n"
                  << "  rows: " << rows << '\n'
                  << "  features: " << features << '\n'
                  << "  trees: " << regression.num_trees() << '\n'
                  << "  prediction ms: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - start).count() << '\n';
            std::cout << "  finite predictions: " << finite_predictions
                  << '/' << rows << '\n'
                  << "  prediction min: " << minimum << '\n'
                  << "  prediction max: " << maximum << '\n'
                  << "  prediction mean: " << mean << '\n'
                  << "  prediction stddev: " << std::sqrt(variance) << '\n';
        for (std::size_t index = 0; index < first_predictions.size(); ++index)
        {
            std::cout << "  prediction[" << index << "]: "
                      << first_predictions[index] << '\n';
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
