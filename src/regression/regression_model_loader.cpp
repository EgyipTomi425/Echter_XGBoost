#include "regression_model.cuh"

#include "../core/cuda_utils.cuh"

#include <cudf/copying.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace echter::xgb::detail
{
namespace
{

// Objectives whose predictions are the raw sum of the base score and the
// leaf values, which is exactly what the prediction kernel computes.
constexpr std::array<std::string_view, 6> identity_objectives{
    "reg:squarederror",
    "reg:linear",
    "reg:squaredlogerror",
    "reg:pseudohubererror",
    "reg:absoluteerror",
    "reg:quantileerror"};

// A column of the parsed model together with its schema entry, which holds the
// JSON field names of struct children.
struct JsonField
{
    cudf::column_view column;
    const cudf::io::column_name_info* schema;
    std::string path;
};

// Host copy of a LIST column: row r is values[offsets[r], offsets[r + 1]).
template <typename T>
struct HostLists
{
    std::vector<cudf::size_type> offsets;
    std::vector<T> values;

    [[nodiscard]] std::span<const T> row(std::size_t index) const
    {
        return {values.data() + offsets[index], values.data() + offsets[index + 1]};
    }
};

struct TreeColumns
{
    HostLists<int> left_children;
    HostLists<int> right_children;
    HostLists<int> split_indices;
    HostLists<float> split_conditions;
    HostLists<std::int8_t> default_left;
    // Older XGBoost versions do not write split types.
    std::optional<HostLists<std::int8_t>> split_type;
};

// cuDF's JSON reader expects records, so the model object is read as the only
// element of a JSON array.
std::string read_as_json_array(const std::string& path)
{
    if (!std::filesystem::is_regular_file(path))
    {
        throw std::runtime_error("file not found");
    }

    const auto source = cudf::io::datasource::create(path);
    const std::size_t size = source->size();
    std::string json(size + 2, '\0');
    const std::size_t bytes = source->host_read(
        0,
        size,
        reinterpret_cast<std::uint8_t*>(json.data() + 1));
    json.resize(bytes + 2);
    json.front() = '[';
    json.back() = ']';
    return json;
}

JsonField root_field(
    const cudf::io::table_with_metadata& table,
    const std::string& name)
{
    const auto view = table.tbl->view();
    const auto& schema = table.metadata.schema_info;
    for (int i = 0; i < view.num_columns() && i < static_cast<int>(schema.size()); ++i)
    {
        if (schema[i].name == name)
        {
            return {view.column(i), &schema[i], name};
        }
    }
    throw std::runtime_error("missing '" + name + "'");
}

std::optional<JsonField> find_field(const JsonField& parent, const std::string& name)
{
    if (parent.column.type().id() != cudf::type_id::STRUCT)
    {
        return std::nullopt;
    }

    const cudf::structs_column_view view(parent.column);
    const auto& children = parent.schema->children;
    for (int i = 0; i < view.num_children() && i < static_cast<int>(children.size()); ++i)
    {
        if (children[i].name == name)
        {
            return JsonField{view.get_sliced_child(i), &children[i], parent.path + "." + name};
        }
    }
    return std::nullopt;
}

JsonField field(const JsonField& parent, const std::string& name)
{
    auto result = find_field(parent, name);
    if (!result)
    {
        throw std::runtime_error("missing '" + parent.path + "." + name + "'");
    }
    return std::move(*result);
}

// cuDF describes a LIST column with the children "offsets" and "element".
JsonField list_elements(const JsonField& list)
{
    if (list.column.type().id() != cudf::type_id::LIST || list.schema->children.empty())
    {
        throw std::runtime_error("'" + list.path + "' is not a list");
    }

    const cudf::lists_column_view view(list.column);
    return {
        view.get_sliced_child(cudf::get_default_stream()),
        &list.schema->children.back(),
        list.path + "[]"};
}

std::string read_string(const JsonField& field)
{
    if (field.column.type().id() != cudf::type_id::STRING || field.column.size() == 0)
    {
        throw std::runtime_error("'" + field.path + "' is not a string");
    }

    const auto scalar = cudf::get_element(field.column, 0);
    if (!scalar->is_valid())
    {
        throw std::runtime_error("'" + field.path + "' is null");
    }
    return static_cast<const cudf::string_scalar&>(*scalar).to_string();
}

template <typename T>
T parse_number(std::string_view text, const std::string& what)
{
    T value{};
    const auto* const end = text.data() + text.size();
    const auto [parsed_end, error] = std::from_chars(text.data(), end, value);
    if (error != std::errc{} || parsed_end != end)
    {
        throw std::runtime_error("invalid " + what + " '" + std::string(text) + "'");
    }
    return value;
}

// XGBoost 2.x and later write the base score as a one-element list, e.g. "[5E-1]".
float parse_base_score(std::string_view text)
{
    if (text.size() >= 2 && text.front() == '[' && text.back() == ']')
    {
        text = text.substr(1, text.size() - 2);
    }
    return parse_number<float>(text, "base_score");
}

int read_int_param(const JsonField& field)
{
    return parse_number<int>(read_string(field), "'" + field.path + "'");
}

// Copies a whole LIST column to the host with one transfer for the offsets and
// one for the values, instead of reading it row by row.
template <typename T>
HostLists<T> copy_lists_to_host(const JsonField& list)
{
    if (list.column.type().id() != cudf::type_id::LIST)
    {
        throw std::runtime_error("'" + list.path + "' is not a list");
    }

    const cudf::lists_column_view view(list.column);
    HostLists<T> result;
    result.offsets.resize(static_cast<std::size_t>(view.size()) + 1);
    if (view.size() == 0)
    {
        return result;
    }
    check_cuda(
        cudaMemcpy(
            result.offsets.data(),
            view.offsets_begin(),
            result.offsets.size() * sizeof(cudf::size_type),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(model list offsets D2H)");

    // The offsets index the unsliced child, while the sliced child starts at
    // the first offset of this view.
    const auto first = result.offsets.front();
    for (auto& offset : result.offsets)
    {
        offset -= first;
    }

    cudf::column_view values = view.get_sliced_child(cudf::get_default_stream());
    std::unique_ptr<cudf::column> converted;
    const cudf::data_type type{cudf::type_to_id<T>()};
    if (values.type() != type)
    {
        converted = cudf::cast(values, type);
        values = converted->view();
    }

    result.values.resize(static_cast<std::size_t>(values.size()));
    if (!result.values.empty())
    {
        check_cuda(
            cudaMemcpy(
                result.values.data(),
                values.data<T>(),
                result.values.size() * sizeof(T),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(model list values D2H)");
    }
    return result;
}

void check_supported_model(const JsonField& learner, const JsonField& params)
{
    if (const auto objective = find_field(learner, "objective"))
    {
        const auto name = read_string(field(*objective, "name"));
        if (std::find(identity_objectives.begin(), identity_objectives.end(), name)
            == identity_objectives.end())
        {
            throw std::runtime_error(
                "unsupported objective '" + name
                + "': only regression objectives without an output transformation are supported");
        }
    }

    if (const auto name = find_field(field(learner, "gradient_booster"), "name");
        name && read_string(*name) != "gbtree")
    {
        throw std::runtime_error("unsupported booster '" + read_string(*name) + "'");
    }

    if (const auto num_target = find_field(params, "num_target");
        num_target && read_int_param(*num_target) != 1)
    {
        throw std::runtime_error("multi-target models are not supported");
    }

    if (const auto num_class = find_field(params, "num_class");
        num_class && read_int_param(*num_class) != 0)
    {
        throw std::runtime_error("classification models are not supported");
    }
}

TreeColumns read_tree_columns(const JsonField& trees)
{
    TreeColumns columns{
        copy_lists_to_host<int>(field(trees, "left_children")),
        copy_lists_to_host<int>(field(trees, "right_children")),
        copy_lists_to_host<int>(field(trees, "split_indices")),
        copy_lists_to_host<float>(field(trees, "split_conditions")),
        copy_lists_to_host<std::int8_t>(field(trees, "default_left")),
        std::nullopt};
    if (const auto split_type = find_field(trees, "split_type"))
    {
        columns.split_type = copy_lists_to_host<std::int8_t>(*split_type);
    }
    return columns;
}

void append_tree(
    const TreeColumns& columns,
    std::size_t tree,
    int num_features,
    HostModel& model)
{
    const auto left = columns.left_children.row(tree);
    const auto right = columns.right_children.row(tree);
    const auto split_indices = columns.split_indices.row(tree);
    const auto split_conditions = columns.split_conditions.row(tree);
    const auto default_left = columns.default_left.row(tree);
    const auto split_type = columns.split_type
        ? columns.split_type->row(tree)
        : std::span<const std::int8_t>{};

    const auto error = [tree](const std::string& message)
    {
        return std::runtime_error("tree " + std::to_string(tree) + ": " + message);
    };

    const std::size_t size = left.size();
    if (size == 0)
    {
        throw error("tree has no nodes");
    }
    if (right.size() != size || split_indices.size() != size
        || split_conditions.size() != size || default_left.size() != size
        || (columns.split_type && split_type.size() != size))
    {
        throw error("node arrays have different lengths");
    }
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()) - model.nodes.size())
    {
        throw std::runtime_error("model has too many nodes");
    }

    const int base = static_cast<int>(model.nodes.size());
    const int tree_size = static_cast<int>(size);
    model.entry_nodes.push_back(base);
    model.nodes.resize(model.nodes.size() + size, Node{});

    // Only nodes reachable from the root are converted and checked; XGBoost may
    // keep unreachable (deleted) nodes in the arrays. Visiting a node twice
    // means the child links do not form a tree, which would make the kernel
    // loop forever.
    std::vector<bool> visited(size, false);
    std::vector<int> pending{0};
    while (!pending.empty())
    {
        const int index = pending.back();
        pending.pop_back();
        if (visited[index])
        {
            throw error("node " + std::to_string(index) + " is reachable more than once");
        }
        visited[index] = true;

        Node& node = model.nodes[base + index];

        // XGBoost marks leaves with a left child of -1 and stores the leaf
        // value in split_conditions.
        if (left[index] == -1)
        {
            node = Node{
                .left = -1,
                .right = -1,
                .split_feature = 0,
                .value = split_conditions[index]};
            continue;
        }

        if (!split_type.empty() && split_type[index] != 0)
        {
            throw error("categorical splits are not supported");
        }
        if (split_indices[index] < 0 || split_indices[index] >= num_features)
        {
            throw error(
                "node " + std::to_string(index) + " splits on feature "
                + std::to_string(split_indices[index]) + ", but the model has "
                + std::to_string(num_features) + " features");
        }
        for (const int child : {left[index], right[index]})
        {
            if (child < 0 || child >= tree_size)
            {
                throw error(
                    "node " + std::to_string(index) + " has invalid child index "
                    + std::to_string(child));
            }
        }

        node = Node{
            .left = base + left[index],
            .right = base + right[index],
            .split_feature = static_cast<std::uint32_t>(split_indices[index])
                | (default_left[index] != 0 ? default_left_flag : 0u),
            .value = split_conditions[index]};
        pending.push_back(right[index]);
        pending.push_back(left[index]);
    }
}

}  // namespace

HostModel load_model_with_cudf(const std::string& json_path)
{
    const std::string json = read_as_json_array(json_path);
    const auto options = cudf::io::json_reader_options::builder(
                             cudf::io::source_info(
                                 cudf::host_span<const char>(json.data(), json.size())))
                             .compression(cudf::io::compression_type::NONE)
                             .lines(false)
                             .experimental(true)
                             .build();
    const auto parsed = cudf::io::read_json(options);
    if (!parsed.tbl || parsed.tbl->num_rows() != 1)
    {
        throw std::runtime_error("model JSON must contain exactly one object");
    }

    const auto learner = root_field(parsed, "learner");
    const auto params = field(learner, "learner_model_param");
    check_supported_model(learner, params);

    HostModel model;
    model.base_score = parse_base_score(read_string(field(params, "base_score")));
    model.num_features = read_int_param(field(params, "num_feature"));
    if (model.num_features <= 0)
    {
        throw std::runtime_error("model has no features");
    }

    const auto trees = list_elements(
        field(field(field(learner, "gradient_booster"), "model"), "trees"));
    const auto columns = read_tree_columns(trees);
    const std::size_t tree_count = columns.left_children.offsets.size() - 1;
    if (tree_count == 0)
    {
        throw std::runtime_error("model has no trees");
    }

    model.entry_nodes.reserve(tree_count);
    model.nodes.reserve(columns.left_children.values.size());
    for (std::size_t tree = 0; tree < tree_count; ++tree)
    {
        append_tree(columns, tree, model.num_features, model);
    }
    return model;
}

}  // namespace echter::xgb::detail
