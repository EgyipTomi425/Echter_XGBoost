#include <algorithm>
#include <charconv>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

import echter.xgb;

namespace
{

std::vector<std::size_t> parse_columns(std::string_view value)
{
    std::vector<std::size_t> columns;
    while (true)
    {
        const std::size_t comma = value.find(',');
        const auto token = value.substr(0, comma);
        std::size_t index = 0;
        const auto [end, error] =
            std::from_chars(token.data(), token.data() + token.size(), index);
        if (token.empty() || error != std::errc{} || end != token.data() + token.size())
        {
            throw std::invalid_argument("invalid column index '" + std::string(token) + "'");
        }
        columns.push_back(index);
        if (comma == std::string_view::npos)
        {
            return columns;
        }
        value.remove_prefix(comma + 1);
    }
}

}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: echter_xgb_csv_prediction <model.json> <input.csv> "
                     "[key-column] [--key-column INDEX] "
                     "[--target-columns INDEX[,INDEX...]] "
                     "[--ignore-columns INDEX[,INDEX...]]\n";
        return 2;
    }

    try
    {
        const std::string model_path = argv[1];
        const std::string input_path = argv[2];
        std::size_t key_column = 0;
        std::vector<std::size_t> target_columns;
        std::vector<std::size_t> excluded_columns;
        int argument = 3;
        if (argument < argc && argv[argument][0] != '-')
        {
            key_column = parse_columns(argv[argument++]).front();
        }
        while (argument + 1 < argc)
        {
            const std::string option = argv[argument];
            const auto columns = parse_columns(argv[argument + 1]);
            argument += 2;
            if (option == "--key-column" && columns.size() == 1)
            {
                key_column = columns.front();
            }
            else if (option == "--target-columns")
            {
                target_columns.insert(target_columns.end(), columns.begin(), columns.end());
            }
            else if (option == "--ignore-columns")
            {
                excluded_columns.insert(excluded_columns.end(), columns.begin(), columns.end());
            }
            else
            {
                throw std::invalid_argument("invalid option: " + option);
            }
        }
        if (argument != argc)
        {
            throw std::invalid_argument("missing value for option " + std::string(argv[argument]));
        }

        echter::xgb::XGBoost api;
        auto& regression = api.regression();
        regression.load_model(model_path);

        // The key, target, and ignored columns are removed from the model input.
        const auto input = echter::xgb::io::read_csv(input_path);
        std::sort(target_columns.begin(), target_columns.end());
        target_columns.erase(
            std::unique(target_columns.begin(), target_columns.end()),
            target_columns.end());
        excluded_columns.insert(excluded_columns.end(), target_columns.begin(), target_columns.end());
        excluded_columns.push_back(key_column);
        std::sort(excluded_columns.begin(), excluded_columns.end());
        excluded_columns.erase(
            std::unique(excluded_columns.begin(), excluded_columns.end()),
            excluded_columns.end());
        const auto features = echter::xgb::io::select_columns(input, excluded_columns);
        const auto predictions = regression.predict(features);

        // The output holds the key, the targets, and the prediction.
        const auto input_view = input.view();
        const auto column_view = [&](std::size_t column)
        {
            return echter::xgb::DeviceColumnarView{
                input_view.data + column * input_view.rows, input_view.rows, 1};
        };
        std::vector<echter::xgb::DeviceColumnarView> output_columns{column_view(key_column)};
        std::vector<std::string> output_names{"id"};
        for (const auto column : target_columns)
        {
            output_columns.push_back(column_view(column));
            output_names.push_back("target_" + std::to_string(column));
        }
        output_columns.push_back(predictions.values.view());
        output_names.push_back("prediction");

        std::filesystem::create_directories("outputs");
        const std::string output_path =
            "outputs/" + std::filesystem::path(input_path).stem().string()
            + "_predictions.csv";
        echter::xgb::io::write_csv(output_path, output_columns, output_names);

        std::cout << "CSV prediction example\n"
                  << "  rows: " << input.rows() << '\n'
                  << "  trees: " << regression.num_trees() << '\n'
                  << "  output: " << output_path << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
