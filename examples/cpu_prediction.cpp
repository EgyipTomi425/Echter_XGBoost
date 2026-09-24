#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

import echter.xgb;

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "usage: echter_xgb_cpu_prediction <model.json> [rows]\n";
        return 2;
    }

    try
    {
        const std::size_t rows = argc == 3 ? std::stoull(argv[2]) : 100000;
        echter::xgb::XGBoost api;
        auto& regression = api.regression();

        const auto model_start = std::chrono::steady_clock::now();
        regression.load_model(argv[1]);
        const auto model_end = std::chrono::steady_clock::now();

        const std::size_t features = regression.num_features();
        std::vector<float> input(rows * features);
        std::mt19937 generator(12345u);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (float& value : input)
        {
            value = distribution(generator);
        }

        const auto prediction_start = std::chrono::steady_clock::now();
        const echter::xgb::HostColumnarView host_input{
            input.data(), rows, features};
        const auto prediction = regression.predict(
            host_input);
        const auto prediction_end = std::chrono::steady_clock::now();

        std::cout << "CPU prediction\n"
                  << "  rows: " << rows << '\n'
                  << "  features: " << features << '\n'
                  << "  trees: " << regression.num_trees() << '\n'
                  << "  model load ms: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         model_end - model_start).count() << '\n'
                  << "  prediction ms: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         prediction_end - prediction_start).count() << '\n';

        const auto count = std::min<std::size_t>(10, prediction.values.size());
        for (std::size_t index = 0; index < count; ++index)
        {
            std::cout << "  prediction[" << index << "]: "
                      << prediction.values[index] << '\n';
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
