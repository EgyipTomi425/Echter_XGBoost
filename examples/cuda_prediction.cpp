#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

import echter.xgb;

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "usage: echter_xgb_cuda_prediction <model.json> [rows]\n";
        return 2;
    }

    try
    {
        const std::size_t rows = argc == 3 ? std::stoull(argv[2]) : 100000;
        echter::xgb::XGBoost api;
        auto& regression = api.regression();
        regression.load_model(argv[1]);

        const std::size_t features = regression.num_features();
        std::vector<float> host_input(rows * features);
        std::mt19937 generator(12345u);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (float& value : host_input)
        {
            value = distribution(generator);
        }

        auto device_input = regression.allocate_device(rows, features);
        if (cudaMemcpy(
                device_input.mutable_data(),
                host_input.data(),
                host_input.size() * sizeof(float),
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            throw std::runtime_error("failed to upload CUDA example input");
        }

        const auto start = std::chrono::steady_clock::now();
        const auto prediction = regression.predict(device_input);
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            throw std::runtime_error("CUDA prediction synchronization failed");
        }
        const auto end = std::chrono::steady_clock::now();

        std::vector<float> first_predictions(std::min<std::size_t>(10, rows));
        if (!first_predictions.empty()
            && cudaMemcpy(
                first_predictions.data(),
                prediction.values.view().data,
                first_predictions.size() * sizeof(float),
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            throw std::runtime_error("failed to read CUDA example output");
        }

        std::cout << "CUDA prediction\n"
                  << "  rows: " << rows << '\n'
                  << "  features: " << features << '\n'
                  << "  trees: " << regression.num_trees() << '\n'
                  << "  prediction ms: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - start).count() << '\n';
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
