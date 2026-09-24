#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

import echter.xgb;

namespace
{

long long elapsed_ms(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::size_t parse_index(std::string_view text)
{
    std::size_t value = 0;
    const auto* const end = text.data() + text.size();
    const auto [parsed_end, error] = std::from_chars(text.data(), end, value);
    if (text.empty() || error != std::errc{} || parsed_end != end)
    {
        throw std::invalid_argument("invalid column index '" + std::string(text) + "'");
    }
    return value;
}

std::vector<std::size_t> parse_columns(std::string_view value)
{
    std::vector<std::size_t> columns;
    while (true)
    {
        const std::size_t comma = value.find(',');
        columns.push_back(parse_index(value.substr(0, comma)));
        if (comma == std::string_view::npos)
        {
            return columns;
        }
        value.remove_prefix(comma + 1);
    }
}

void sort_unique(std::vector<std::size_t>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: echter_xgb_predict <model.json> <input.csv> "
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
            key_column = parse_index(argv[argument++]);
        }
        while (argument < argc)
        {
            const std::string option = argv[argument++];
            if (argument >= argc)
            {
                throw std::invalid_argument("missing value for option " + option);
            }
            const auto columns = parse_columns(argv[argument++]);
            if (option == "--key-column")
            {
                if (columns.size() != 1)
                {
                    throw std::invalid_argument("--key-column expects one index");
                }
                key_column = columns.front();
            }
            else if (option == "--target-columns")
            {
                target_columns.insert(target_columns.end(), columns.begin(), columns.end());
                excluded_columns.insert(excluded_columns.end(), columns.begin(), columns.end());
            }
            else if (option == "--ignore-columns")
            {
                excluded_columns.insert(excluded_columns.end(), columns.begin(), columns.end());
            }
            else
            {
                throw std::invalid_argument("unknown option: " + option);
            }
        }
        const auto total_start = std::chrono::steady_clock::now();

        echter::xgb::XGBoost api;
        auto& regression = api.regression();

        const auto model_start = std::chrono::steady_clock::now();
        regression.load_model(model_path);
        const auto model_end = std::chrono::steady_clock::now();

        const auto read_start = std::chrono::steady_clock::now();
        const auto input = echter::xgb::io::read_csv(input_path);
        const auto read_end = std::chrono::steady_clock::now();

        excluded_columns.push_back(key_column);
        sort_unique(target_columns);
        sort_unique(excluded_columns);
        if (excluded_columns.back() >= input.features())
        {
            throw std::runtime_error(
                "column index " + std::to_string(excluded_columns.back())
                + " is out of range: the CSV has " + std::to_string(input.features())
                + " columns");
        }
        if (input.features() - excluded_columns.size() != regression.num_features())
        {
            throw std::runtime_error(
                "CSV feature count does not match model: model expects "
                + std::to_string(regression.num_features())
                + ", input has "
                + std::to_string(input.features() - excluded_columns.size()));
        }

        const auto input_view = input.view();
        const auto features =
            echter::xgb::io::select_columns(input, excluded_columns);
        const auto predict_start = std::chrono::steady_clock::now();
        const auto predictions = regression.predict(features);
        const auto predict_end = std::chrono::steady_clock::now();

        std::filesystem::create_directories("outputs");
        const std::string output_path =
            "outputs/" + std::filesystem::path(input_path).stem().string()
            + "_predictions.csv";
        const auto write_start = std::chrono::steady_clock::now();
        std::vector<echter::xgb::DeviceColumnarView> output_columns{
            {input_view.data + key_column * input_view.rows, input_view.rows, 1}};
        std::vector<std::string> output_names{"id"};
        for (const auto column : target_columns)
        {
            output_columns.push_back(
                {input_view.data + column * input_view.rows, input_view.rows, 1});
            output_names.push_back("target_" + std::to_string(column));
        }
        output_columns.push_back(predictions.values.view());
        output_names.push_back("prediction");
        echter::xgb::io::write_csv(
            output_path,
            output_columns,
            output_names);
        const auto write_end = std::chrono::steady_clock::now();

        std::cout << "Echter XGBoost prediction\n"
                  << "  model: " << model_path << '\n'
                  << "  input: " << input_path << '\n'
                  << "  output: " << output_path << '\n'
                  << "  rows: " << input.rows() << '\n'
                  << "  input columns: " << input.features() << '\n'
                  << "  feature columns: " << regression.num_features() << '\n'
                  << "  trees: " << regression.num_trees() << '\n'
                  << "  model load ms: " << elapsed_ms(model_start, model_end) << '\n'
                  << "  CSV read ms: " << elapsed_ms(read_start, read_end) << '\n'
                  << "  prediction ms: " << elapsed_ms(predict_start, predict_end) << '\n'
                  << "  CSV write ms: " << elapsed_ms(write_start, write_end) << '\n'
                  << "  total ms: " << elapsed_ms(total_start, write_end) << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
