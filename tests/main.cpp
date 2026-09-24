#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

import echter.xgb;

namespace
{

std::vector<float> make_random_data(
    std::size_t rows,
    std::size_t features)
{
    std::mt19937 rng(123456u);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> data(rows * features);

    for (std::size_t feature = 0; feature < features; ++feature)
    {
        for (std::size_t row = 0; row < rows; ++row)
        {
            data[feature * rows + row] = distribution(rng);
        }
    }

    return data;
}

}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "usage: echter_xgb_test <model.json> [rows]\n";
        return 2;
    }

    try
    {
        const std::size_t rows = argc == 3 ? std::stoull(argv[2]) : 8;

        echter::xgb::XGBoost api;
        auto& regression = api.regression();
        regression.load_model(argv[1]);

        const std::size_t features = regression.num_features();
        const auto host = make_random_data(rows, features);
        const echter::xgb::HostColumnarView host_view{
            host.data(),
            rows,
            features};

        const auto device = regression.upload(host_view);
        const auto prediction_a = regression.predict(device.view());
        const auto prediction_b = regression.predict(host_view);

        assert(prediction_a.values.rows() == rows);
        assert(prediction_b.values.size() == rows);

        for (std::size_t row = 0; row < rows; ++row)
        {
            assert(std::isfinite(prediction_b.values[row]));
        }

        std::cout << "Echter XGBoost test passed\n";
        std::cout << "  model features: " << features << '\n';
        std::cout << "  trees: " << regression.num_trees() << '\n';
        std::cout << "  rows: " << rows << '\n';
        std::cout << "  layout: data[feature * rows + row]\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "test error: " << error.what() << '\n';
        return 1;
    }
}
