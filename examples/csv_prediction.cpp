#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

import echter.xgb;

namespace
{

std::string file_stem(const std::string& path)
{
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    const std::size_t end =
        dot == std::string::npos || (slash != std::string::npos && dot < slash)
            ? path.size()
            : dot;
    const std::size_t start = slash == std::string::npos ? 0 : slash + 1;
    return path.substr(start, end - start);
}

void ensure_output_directory()
{
    struct stat info{};
    if ((stat("outputs", &info) == 0 && S_ISDIR(info.st_mode))
        || mkdir("outputs", 0755) == 0
        || errno == EEXIST)
    {
        return;
    }
    throw std::runtime_error(
        std::string("cannot create outputs directory: ") + std::strerror(errno));
}

long long elapsed_ms(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::vector<std::size_t> parse_columns(const std::string& value)
{
    std::vector<std::size_t> columns;
    std::size_t start = 0;
    while (start < value.size())
    {
        const std::size_t end = value.find(',', start);
        const std::string token = value.substr(
            start,
            end == std::string::npos ? std::string::npos : end - start);
        if (token.empty())
        {
            throw std::invalid_argument("empty column index");
        }
        columns.push_back(std::stoull(token));
        if (end == std::string::npos)
        {
            break;
        }
        start = end + 1;
    }
    return columns;
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
            key_column = std::stoull(argv[argument++]);
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
                target_columns.insert(
                    target_columns.end(), columns.begin(), columns.end());
                excluded_columns.insert(
                    excluded_columns.end(), columns.begin(), columns.end());
            }
            else if (option == "--ignore-columns")
            {
                excluded_columns.insert(
                    excluded_columns.end(), columns.begin(), columns.end());
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

        if (input.features() < 2 || key_column >= input.features())
        {
            throw std::runtime_error("invalid key column or CSV has no feature columns");
        }
        for (const auto column : target_columns)
        {
            if (column >= input.features())
            {
                throw std::runtime_error("target column is out of range");
            }
        }
        std::sort(target_columns.begin(), target_columns.end());
        target_columns.erase(
            std::unique(target_columns.begin(), target_columns.end()),
            target_columns.end());
        excluded_columns.push_back(key_column);
        std::sort(excluded_columns.begin(), excluded_columns.end());
        excluded_columns.erase(
            std::unique(excluded_columns.begin(), excluded_columns.end()),
            excluded_columns.end());
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

        ensure_output_directory();
        const std::string output_path =
            "outputs/" + file_stem(input_path) + "_predictions.csv";
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

        std::cout << "CSV prediction example\n"
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
