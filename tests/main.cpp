#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

import echter.xgb;

namespace
{

int checks = 0;
int failures = 0;

void check(bool condition, const std::string& description)
{
    ++checks;
    if (!condition)
    {
        ++failures;
        std::cerr << "FAILED: " << description << '\n';
    }
}

template <typename Action>
void check_throws(Action&& action, const std::string& expected, const std::string& description)
{
    try
    {
        action();
    }
    catch (const std::exception& error)
    {
        const std::string message = error.what();
        check(
            message.find(expected) != std::string::npos,
            description + " (message: " + message + ")");
        return;
    }
    check(false, description + " (no exception)");
}

bool same_values(const std::vector<float>& actual, const std::vector<float>& expected)
{
    if (actual.size() != expected.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        const bool both_nan = std::isnan(actual[index]) && std::isnan(expected[index]);
        if (!both_nan && actual[index] != expected[index])
        {
            return false;
        }
    }
    return true;
}

std::vector<float> download(const echter::xgb::DeviceColumnarView& view)
{
    std::vector<float> host(view.rows * view.features);
    if (!host.empty()
        && cudaMemcpy(
               host.data(),
               view.data,
               host.size() * sizeof(float),
               cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        throw std::runtime_error("failed to copy test data to host");
    }
    return host;
}

// A file in the temporary directory that is removed at scope exit.
class TempFile
{
public:
    TempFile(const std::string& extension, const std::string& contents = {})
        : path_(std::filesystem::temp_directory_path()
                / ("echter_xgb_test_" + std::to_string(std::random_device{}()) + extension))
    {
        if (!contents.empty())
        {
            std::ofstream(path_) << contents;
        }
    }

    ~TempFile()
    {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    [[nodiscard]] std::string path() const
    {
        return path_.string();
    }

private:
    std::filesystem::path path_;
};

struct TreeArrays
{
    std::string left_children;
    std::string right_children;
    std::string split_indices;
    std::string split_conditions;
    std::string default_left;
};

std::string tree_json(std::size_t id, const TreeArrays& tree)
{
    const auto nodes = std::count(tree.left_children.begin(), tree.left_children.end(), ',') + 1;
    // base_weights deliberately differ from the leaf values: XGBoost predicts
    // with the leaf values stored in split_conditions.
    std::string base_weights = "[";
    std::string split_type = "[";
    for (long node = 0; node < nodes; ++node)
    {
        base_weights += node == 0 ? "9.9E1" : ",9.9E1";
        split_type += node == 0 ? "0" : ",0";
    }
    base_weights += "]";
    split_type += "]";

    return "{\"base_weights\":" + base_weights
        + ",\"default_left\":" + tree.default_left
        + ",\"id\":" + std::to_string(id)
        + ",\"left_children\":" + tree.left_children
        + ",\"right_children\":" + tree.right_children
        + ",\"split_conditions\":" + tree.split_conditions
        + ",\"split_indices\":" + tree.split_indices
        + ",\"split_type\":" + split_type
        + ",\"tree_param\":{\"num_deleted\":\"0\",\"num_feature\":\"2\",\"num_nodes\":\""
        + std::to_string(nodes) + "\",\"size_leaf_vector\":\"1\"}}";
}

// A two-feature model in XGBoost's JSON format with base score 0.5:
//   tree 0: x0 < 0.5 ? 1 : -1, missing x0 goes left
//   tree 1: x1 < 2 ? (x0 < -1 ? 10 : 20) : 0.25, missing x1 goes right,
//           missing x0 goes left; the right child precedes the left one
const TreeArrays tree_0{"[1,-1,-1]", "[2,-1,-1]", "[0,0,0]", "[5E-1,1E0,-1E0]", "[1,0,0]"};
const TreeArrays tree_1{
    "[2,-1,3,-1,-1]",
    "[1,-1,4,-1,-1]",
    "[1,0,0,0,0]",
    "[2E0,2.5E-1,-1E0,1E1,2E1]",
    "[0,0,1,0,0]"};

std::string model_json(
    const std::vector<TreeArrays>& trees = {tree_0, tree_1},
    const std::string& objective = "reg:squarederror",
    const std::string& base_score = "5E-1")
{
    std::string tree_list;
    std::string iteration_indptr = "0";
    std::string tree_info;
    for (std::size_t tree = 0; tree < trees.size(); ++tree)
    {
        tree_list += (tree == 0 ? "" : ",") + tree_json(tree, trees[tree]);
        iteration_indptr += "," + std::to_string(tree + 1);
        tree_info += tree == 0 ? "0" : ",0";
    }

    return "{\"learner\":{\"gradient_booster\":{\"model\":{"
           "\"gbtree_model_param\":{\"num_parallel_tree\":\"1\",\"num_trees\":\""
        + std::to_string(trees.size()) + "\"},\"iteration_indptr\":[" + iteration_indptr
        + "],\"tree_info\":[" + tree_info + "],\"trees\":[" + tree_list
        + "]},\"name\":\"gbtree\"},"
          "\"learner_model_param\":{\"base_score\":\"[" + base_score
        + "]\",\"boost_from_average\":\"1\","
          "\"num_class\":\"0\",\"num_feature\":\"2\",\"num_target\":\"1\"},"
          "\"objective\":{\"name\":\""
        + objective + "\",\"reg_loss_param\":{\"scale_pos_weight\":\"1\"}}},"
                      "\"version\":[3,2,0]}";
}

const float nan = std::numeric_limits<float>::quiet_NaN();

// Column-major input for the test model and the expected predictions.
const std::vector<float> test_input{
    0.0f, 1.0f, nan, -2.0f, 0.5f,   // x0
    0.0f, 3.0f, 1.0f, nan, 2.0f};   // x1
const std::vector<float> expected_predictions{21.5f, -0.25f, 11.5f, 1.75f, -0.25f};
constexpr std::size_t test_rows = 5;

void test_prediction()
{
    const TempFile model_file(".json", model_json());
    echter::xgb::XGBoost api;
    auto& regression = api.regression();

    check_throws(
        [&] { (void)regression.predict(echter::xgb::HostColumnarView{test_input.data(), test_rows, 2}); },
        "not loaded",
        "predicting without a model throws");

    regression.load_model(model_file.path());
    check(regression.num_features() == 2, "model has 2 features");
    check(regression.num_trees() == 2, "model has 2 trees");

    const echter::xgb::HostColumnarView host{test_input.data(), test_rows, 2};
    const auto host_prediction = regression.predict(host);
    check(
        same_values(host_prediction.values, expected_predictions),
        "host predictions match hand-computed values, including missing values");

    const auto device_input = regression.upload(host);
    const auto device_prediction = regression.predict(device_input);
    check(device_prediction.values.rows() == test_rows, "device prediction has one row per input");
    check(
        same_values(download(device_prediction.values.view()), expected_predictions),
        "device predictions match host predictions");

    const auto empty = regression.predict(echter::xgb::HostColumnarView{nullptr, 0, 2});
    check(empty.values.empty(), "empty input gives empty prediction");

    check_throws(
        [&] { (void)regression.predict(echter::xgb::HostColumnarView{test_input.data(), 2, 5}); },
        "feature count does not match",
        "wrong feature count throws");
}

void test_summation_order()
{
    // Two single-leaf trees of 2^-24 on top of a base score of 1. XGBoost adds
    // the tree sum 2^-23 to the base score and gets 1 + 2^-23; adding each leaf
    // to the base score in turn would round back to exactly 1 both times.
    const TreeArrays leaf{"[-1]", "[-1]", "[0]", "[5.9604645E-8]", "[0]"};
    const TempFile model_file(".json", model_json({leaf, leaf}, "reg:squarederror", "1E0"));
    echter::xgb::XGBoost api;
    auto& regression = api.regression();
    regression.load_model(model_file.path());

    const std::vector<float> input{0.0f, 0.0f};
    const auto prediction = regression.predict(echter::xgb::HostColumnarView{input.data(), 1, 2});
    check(
        prediction.values.size() == 1 && prediction.values[0] == 1.0f + std::ldexp(1.0f, -23),
        "leaf values are summed before the base score is added, like in XGBoost");
}

void test_invalid_models()
{
    echter::xgb::XGBoost api;
    auto& regression = api.regression();

    check_throws(
        [&] { regression.load_model("/nonexistent/model.json"); },
        "failed to load XGBoost regression model",
        "missing model file throws");

    const auto expect_load_error =
        [&](const std::string& json, const std::string& expected, const std::string& description)
    {
        const TempFile model_file(".json", json);
        check_throws([&] { regression.load_model(model_file.path()); }, expected, description);
    };

    expect_load_error(
        model_json({tree_0, tree_1}, "reg:logistic"),
        "unsupported objective 'reg:logistic'",
        "objectives with an output transformation are rejected");
    expect_load_error(
        model_json({tree_0, {"[2,-1,7,-1,-1]", tree_1.right_children, tree_1.split_indices,
                             tree_1.split_conditions, tree_1.default_left}}),
        "invalid child index 7",
        "out-of-range child index is rejected");
    expect_load_error(
        model_json({tree_0, {"[2,-1,0,-1,-1]", tree_1.right_children, tree_1.split_indices,
                             tree_1.split_conditions, tree_1.default_left}}),
        "reachable more than once",
        "cyclic tree is rejected");
    expect_load_error(
        model_json({tree_0, {tree_1.left_children, tree_1.right_children, "[1,0,5,0,0]",
                             tree_1.split_conditions, tree_1.default_left}}),
        "splits on feature 5",
        "out-of-range split feature is rejected");

    const TempFile valid_model(".json", model_json());
    regression.load_model(valid_model.path());
    const TempFile invalid_model(".json", model_json({tree_0, tree_1}, "reg:logistic"));
    check_throws(
        [&] { regression.load_model(invalid_model.path()); },
        "unsupported objective",
        "loading an invalid model throws");
    check(
        same_values(
            regression.predict(echter::xgb::HostColumnarView{test_input.data(), test_rows, 2}).values,
            expected_predictions),
        "a failed load keeps the previously loaded model");
}

void test_csv_io()
{
    const TempFile model_file(".json", model_json());
    echter::xgb::XGBoost api;
    auto& regression = api.regression();
    regression.load_model(model_file.path());

    // Empty fields are missing values.
    const TempFile input_file(
        ".csv",
        "id,x0,x1,target\n"
        "10,0,0,1\n"
        "11,1,3,2\n"
        "12,,1,3\n"
        "13,-2,,4\n"
        "14,0.5,2,5\n");
    const auto table = echter::xgb::io::read_csv(input_file.path());
    check(table.rows() == test_rows && table.features() == 4, "CSV has 5 rows and 4 columns");

    const auto features = echter::xgb::io::select_columns(table, {0, 3});
    check(same_values(download(features.view()), test_input), "selected CSV columns match the input");
    check(
        same_values(download(regression.predict(features).values.view()), expected_predictions),
        "CSV predictions match hand-computed values");

    const TempFile output_file(".csv");
    echter::xgb::io::write_csv(output_file.path(), features, {"x0", "x1"});
    const auto written = echter::xgb::io::read_csv(output_file.path());
    check(
        written.features() == 2 && same_values(download(written.view()), test_input),
        "multi-column table survives a CSV round trip");

    check_throws(
        [&] { echter::xgb::io::write_csv(output_file.path(), features, {"only_one"}); },
        "2 columns",
        "header name count must match the column count");
    check_throws(
        [&] { (void)echter::xgb::io::select_columns(table, {0, 9}); },
        "out of range",
        "out-of-range excluded column throws");
    check_throws(
        [&] { (void)echter::xgb::io::read_csv("/nonexistent/input.csv"); },
        "failed to read CSV",
        "missing CSV file throws");
}

bool pageable_memory_accessible()
{
    int device = 0;
    int accessible = 0;
    return cudaGetDevice(&device) == cudaSuccess
        && cudaDeviceGetAttribute(&accessible, cudaDevAttrPageableMemoryAccess, device) == cudaSuccess
        && accessible != 0;
}

void test_memory_safety()
{
    const TempFile model_file(".json", model_json());
    echter::xgb::Regression regression;
    regression.load_model(model_file.path());
    const echter::xgb::HostColumnarView host{test_input.data(), test_rows, 2};

    auto moved = std::move(regression);
    check(moved.num_trees() == 2, "moved-to model keeps its trees");
    check(
        regression.num_trees() == 0 && regression.num_features() == 0,
        "moved-from model is empty");
    check_throws(
        [&] { (void)regression.predict(host); },
        "not loaded",
        "moved-from model cannot predict");

    auto data = moved.upload(host);
    const auto other = std::move(data);
    check(
        data.rows() == 0 && data.features() == 0 && data.view().data == nullptr,
        "moved-from device data is empty");
    check(same_values(download(other.view()), test_input), "moved-to device data keeps its values");

    if (!pageable_memory_accessible())
    {
        check_throws(
            [&] { (void)moved.predict(echter::xgb::DeviceColumnarView{test_input.data(), test_rows, 2}); },
            "host memory",
            "host data passed as device data is rejected before the kernel runs");
        const TempFile output_file(".csv");
        check_throws(
            [&]
            {
                echter::xgb::io::write_csv(
                    output_file.path(),
                    std::vector<echter::xgb::DeviceColumnarView>{{test_input.data(), test_rows, 1}},
                    {"x0"});
            },
            "host memory",
            "host data passed to write_csv is rejected");
    }
    check(
        same_values(moved.predict(host).values, expected_predictions),
        "the CUDA context still works after rejected inputs");
}

void test_model_file(const std::string& path, std::size_t rows)
{
    echter::xgb::XGBoost api;
    auto& regression = api.regression();
    regression.load_model(path);

    const std::size_t features = regression.num_features();
    std::mt19937 rng(123456u);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> host(rows * features);
    for (float& value : host)
    {
        value = distribution(rng);
    }

    const echter::xgb::HostColumnarView host_view{host.data(), rows, features};
    const auto device_prediction = regression.predict(regression.upload(host_view));
    const auto host_prediction = regression.predict(host_view);
    check(host_prediction.values.size() == rows, "model file: one prediction per row");
    check(
        same_values(download(device_prediction.values.view()), host_prediction.values),
        "model file: device and host predictions agree");

    bool all_finite = true;
    for (const float value : host_prediction.values)
    {
        all_finite = all_finite && std::isfinite(value);
    }
    check(all_finite, "model file: predictions are finite");

    std::cout << "  model file: " << path << " (" << features << " features, "
              << regression.num_trees() << " trees, " << rows << " rows)\n";
}

}

int main(int argc, char** argv)
{
    if (argc > 3)
    {
        std::cerr << "usage: echter_xgb_test [model.json [rows]]\n";
        return 2;
    }

    try
    {
        test_prediction();
        test_summation_order();
        test_invalid_models();
        test_csv_io();
        test_memory_safety();
        if (argc >= 2)
        {
            test_model_file(argv[1], argc == 3 ? std::stoull(argv[2]) : 1000);
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "test error: " << error.what() << '\n';
        return 1;
    }

    if (failures != 0)
    {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "Echter XGBoost tests passed (" << checks << " checks)\n";
    return 0;
}
