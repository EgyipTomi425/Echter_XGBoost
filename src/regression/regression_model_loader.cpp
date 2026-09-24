#include "regression_model.cuh"

#include <cuda_runtime.h>

#include <cudf/copying.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace echter::xgb::detail
{
namespace
{

std::string trim(std::string s)
{
  auto not_space = [](unsigned char c)
{ return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

float parse_xgb_base_score(const std::string& raw)
{
  std::string s = trim(raw);
  if (!s.empty() && s.front() == '[' && s.back() == ']')
{
    s = s.substr(1, s.size() - 2);
  }
  return std::stof(s);
}

bool get_top_level_column(cudf::io::table_with_metadata const& in,
                          std::string const& name,
                          cudf::column_view& out_col,
                          cudf::io::column_name_info const*& out_meta)
{
  auto const view = in.tbl->view();
  auto const& schema = in.metadata.schema_info;
  for (int i = 0; i < view.num_columns() && i < static_cast<int>(schema.size()); ++i)
{
    if (schema[i].name == name)
{
      out_col = view.column(i);
      out_meta = &schema[i];
      return true;
    }
  }
  return false;
}

bool get_struct_child_by_name(cudf::column_view const& struct_col,
                              cudf::io::column_name_info const& struct_meta,
                              std::string const& child_name,
                              cudf::column_view& out_col,
                              cudf::io::column_name_info const*& out_meta)
{
  if (struct_col.type().id() != cudf::type_id::STRUCT) return false;

  cudf::structs_column_view scv(struct_col);
  for (int i = 0; i < scv.num_children() && i < static_cast<int>(struct_meta.children.size()); ++i)
{
    if (struct_meta.children[i].name == child_name)
{
      out_col = scv.get_sliced_child(i);
      out_meta = &struct_meta.children[i];
      return true;
    }
  }
  return false;
}

void print_child_names(char const* label, cudf::io::column_name_info const& meta)
{
  std::cerr << "[xgb] " << label << " children=";
  for (size_t i = 0; i < meta.children.size(); ++i)
{
    if (i) std::cerr << ",";
    std::cerr << meta.children[i].name;
  }
  std::cerr << "\n";
}

bool get_list_row_range(cudf::column_view const& list_col,
                        cudf::size_type row,
                        cudf::size_type& start,
                        cudf::size_type& end)
{
  if (list_col.type().id() != cudf::type_id::LIST) return false;

  cudf::lists_column_view lcv(list_col);
  auto offs = lcv.offsets();
  auto const* d_offs = offs.data<cudf::size_type>();
  if (d_offs == nullptr) return false;

  cudaError_t err = cudaMemcpy(&start, d_offs + row, sizeof(cudf::size_type), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return false;

  err = cudaMemcpy(&end, d_offs + row + 1, sizeof(cudf::size_type), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return false;

  return true;
}

template <typename T>
bool read_list_row_fixed_width(cudf::column_view const& list_col,
                               cudf::size_type row,
                               cudf::data_type target_type,
                               std::vector<T>& out)
{
  if (list_col.type().id() != cudf::type_id::LIST) return false;

  cudf::size_type start = 0;
  cudf::size_type end = 0;
  if (!get_list_row_range(list_col, row, start, end)) return false;

  auto lcv = cudf::lists_column_view(list_col);
  auto child = lcv.get_sliced_child(cudf::get_default_stream());

  std::unique_ptr<cudf::column> casted_child;
  cudf::column_view value_col = child;
  if (child.type().id() != target_type.id())
{
    casted_child = cudf::cast(child, target_type);
    value_col = casted_child->view();
  }

  auto n = static_cast<size_t>(end - start);
  out.resize(n);
  if (n == 0) return true;

  auto const* d_ptr = value_col.data<T>();
  if (d_ptr == nullptr) return false;

  cudaError_t err = cudaMemcpy(out.data(), d_ptr + start, n * sizeof(T), cudaMemcpyDeviceToHost);
  return err == cudaSuccess;
}

bool read_list_row_strings(cudf::column_view const& list_col, cudf::size_type row, std::vector<std::string>& out)
{
  if (list_col.type().id() != cudf::type_id::LIST) return false;

  cudf::size_type start = 0;
  cudf::size_type end = 0;
  if (!get_list_row_range(list_col, row, start, end)) return false;

  auto lcv = cudf::lists_column_view(list_col);
  auto child = lcv.get_sliced_child(cudf::get_default_stream());

  out.clear();
  out.reserve(static_cast<size_t>(end - start));
  for (cudf::size_type i = start; i < end; ++i)
{
    auto s = cudf::get_element(child, i);
    auto* ss = dynamic_cast<cudf::string_scalar*>(s.get());
    if (ss == nullptr) return false;
    out.push_back(ss->to_string());
  }
  return true;
}

bool read_string_scalar_at(cudf::column_view const& col, cudf::size_type row, std::string& out)
{
  auto s = cudf::get_element(col, row);
  auto* ss = dynamic_cast<cudf::string_scalar*>(s.get());
  if (ss == nullptr) return false;
  out = ss->to_string();
  return true;
}

}  // namespace

bool load_model_with_cudf(const std::string& json_path, HostModel& model)
{
  auto src = cudf::io::datasource::create(json_path);
  if (!src)
{
    std::cerr << "[xgb] cannot open model JSON datasource: " << json_path << "\n";
    return false;
  }

  std::string raw(src->size(), '\0');
  auto nread = src->host_read(0, src->size(), reinterpret_cast<uint8_t*>(raw.data()));
  raw.resize(nread);
  if (raw.empty())
{
    std::cerr << "[xgb] model JSON is empty: " << json_path << "\n";
    return false;
  }

  // cuDF JSON reader expects records-like input; wrap model object into a one-row array.
  std::string wrapped;
  wrapped.reserve(raw.size() + 2);
  wrapped.push_back('[');
  wrapped.append(raw);
  wrapped.push_back(']');

  auto host_buf = cudf::host_span<char const>(wrapped.data(), wrapped.size());
  auto options = cudf::io::json_reader_options::builder(cudf::io::source_info(host_buf))
                   .compression(cudf::io::compression_type::NONE)
                   .lines(false)
                   .experimental(true)
                   .build();

  auto parsed = cudf::io::read_json(std::move(options));
  if (!parsed.tbl || parsed.tbl->num_rows() <= 0)
{
    std::cerr << "[xgb] cuDF JSON reader produced empty table for: " << json_path << "\n";
    return false;
  }

  cudf::column_view learner_col;
  cudf::io::column_name_info const* learner_meta = nullptr;
  if (!get_top_level_column(parsed, "learner", learner_col, learner_meta))
{
    std::cerr << "[xgb] missing 'learner' in model JSON" << "\n";
    return false;
  }

  cudf::column_view learner_model_param_col;
  cudf::io::column_name_info const* learner_model_param_meta = nullptr;
  if (!get_struct_child_by_name(
        learner_col, *learner_meta, "learner_model_param", learner_model_param_col, learner_model_param_meta))
{
    std::cerr << "[xgb] missing 'learner_model_param'" << "\n";
    return false;
  }

  cudf::column_view base_score_col;
  cudf::io::column_name_info const* base_score_meta = nullptr;
  if (!get_struct_child_by_name(
        learner_model_param_col, *learner_model_param_meta, "base_score", base_score_col, base_score_meta))
{
    std::cerr << "[xgb] missing 'base_score'" << "\n";
    return false;
  }

  cudf::column_view num_feature_col;
  cudf::io::column_name_info const* num_feature_meta = nullptr;
  if (!get_struct_child_by_name(
        learner_model_param_col, *learner_model_param_meta, "num_feature", num_feature_col, num_feature_meta))
{
    std::cerr << "[xgb] missing 'num_feature'" << "\n";
    return false;
  }

  std::string base_score_raw;
  std::string num_feature_raw;
  if (!read_string_scalar_at(base_score_col, 0, base_score_raw))
{
    std::cerr << "[xgb] failed to read base_score" << "\n";
    return false;
  }
  if (!read_string_scalar_at(num_feature_col, 0, num_feature_raw))
{
    std::cerr << "[xgb] failed to read num_feature" << "\n";
    return false;
  }

  model.base_score = parse_xgb_base_score(base_score_raw);
  model.num_features = std::stoi(num_feature_raw);

  cudf::column_view feature_names_col;
  cudf::io::column_name_info const* feature_names_meta = nullptr;
  if (!get_struct_child_by_name(
        learner_col, *learner_meta, "feature_names", feature_names_col, feature_names_meta))
{
    std::cerr << "[xgb] missing 'feature_names'" << "\n";
    return false;
  }

  model.feature_names.clear();
  if (!read_list_row_strings(feature_names_col, 0, model.feature_names))
{
    std::cerr << "[xgb] failed to read feature_names" << "\n";
    return false;
  }

  cudf::column_view gradient_booster_col;
  cudf::io::column_name_info const* gradient_booster_meta = nullptr;
  if (!get_struct_child_by_name(
        learner_col, *learner_meta, "gradient_booster", gradient_booster_col, gradient_booster_meta))
{
    std::cerr << "[xgb] missing 'gradient_booster'" << "\n";
    return false;
  }

  cudf::column_view gb_model_col;
  cudf::io::column_name_info const* gb_model_meta = nullptr;
  if (!get_struct_child_by_name(
        gradient_booster_col, *gradient_booster_meta, "model", gb_model_col, gb_model_meta))
{
    std::cerr << "[xgb] missing 'gradient_booster.model'" << "\n";
    return false;
  }

  cudf::column_view trees_col;
  cudf::io::column_name_info const* trees_meta = nullptr;
  if (!get_struct_child_by_name(gb_model_col, *gb_model_meta, "trees", trees_col, trees_meta))
{
    std::cerr << "[xgb] missing 'trees'" << "\n";
    return false;
  }

  cudf::size_type trees_start = 0;
  cudf::size_type trees_end = 0;
  if (!get_list_row_range(trees_col, 0, trees_start, trees_end))
{
    std::cerr << "[xgb] failed to read trees list range" << "\n";
    return false;
  }

  auto trees_lcv = cudf::lists_column_view(trees_col);
  auto tree_structs = cudf::structs_column_view(trees_lcv.get_sliced_child(cudf::get_default_stream()));

  cudf::column_view left_children_col;
  cudf::column_view right_children_col;
  cudf::column_view split_conditions_col;
  cudf::column_view split_indices_col;
  cudf::column_view base_weights_col;
  cudf::column_view default_left_col;

  cudf::io::column_name_info const* tree_struct_meta = nullptr;
  if (trees_meta != nullptr)
{
    if (trees_meta->children.size() == 2 && trees_meta->children[1].name == "element")
{
      tree_struct_meta = &trees_meta->children[1];
    } else if (trees_meta->children.size() == 1 && !trees_meta->children[0].children.empty())
{
      tree_struct_meta = &trees_meta->children[0];
    } else if (!trees_meta->children.empty())
{
      tree_struct_meta = trees_meta;
    }
  }

  if (tree_struct_meta == nullptr)
{
    std::cerr << "[xgb] trees element schema missing" << "\n";
    return false;
  }

  cudf::io::column_name_info const* tmp_meta = nullptr;
  if (!get_struct_child_by_name(tree_structs.parent(), *tree_struct_meta, "left_children", left_children_col, tmp_meta))
{
    std::cerr << "[xgb] missing tree.left_children" << "\n";
    print_child_names("tree_struct_meta", *tree_struct_meta);
    if (trees_meta != nullptr) print_child_names("trees_meta", *trees_meta);
    return false;
  }
  if (!get_struct_child_by_name(tree_structs.parent(), *tree_struct_meta, "right_children", right_children_col, tmp_meta))
{
    std::cerr << "[xgb] missing tree.right_children" << "\n";
    return false;
  }
  if (!get_struct_child_by_name(tree_structs.parent(), *tree_struct_meta, "split_conditions", split_conditions_col, tmp_meta))
{
    std::cerr << "[xgb] missing tree.split_conditions" << "\n";
    return false;
  }
  if (!get_struct_child_by_name(tree_structs.parent(), *tree_struct_meta, "split_indices", split_indices_col, tmp_meta))
{
    std::cerr << "[xgb] missing tree.split_indices" << "\n";
    return false;
  }
  if (!get_struct_child_by_name(tree_structs.parent(), *tree_struct_meta, "base_weights", base_weights_col, tmp_meta))
{
    std::cerr << "[xgb] missing tree.base_weights" << "\n";
    return false;
  }
  if (!get_struct_child_by_name(tree_structs.parent(), *tree_struct_meta, "default_left", default_left_col, tmp_meta))
{
    std::cerr << "[xgb] missing tree.default_left" << "\n";
    return false;
  }

  auto num_trees = static_cast<size_t>(trees_end - trees_start);
  model.nodes.clear();
  model.entry_nodes.clear();
  model.entry_nodes.reserve(num_trees);

  std::vector<int> lefts;
  std::vector<int> rights;
  std::vector<float> thresholds;
  std::vector<int> features;
  std::vector<float> leaf_values;
  std::vector<int8_t> defaults;

  for (size_t t = 0; t < num_trees; ++t)
{
    cudf::size_type tree_row = static_cast<cudf::size_type>(trees_start + static_cast<cudf::size_type>(t));

    if (!read_list_row_fixed_width<int>(left_children_col, tree_row, cudf::data_type{cudf::type_id::INT32}, lefts)) return false;
    if (!read_list_row_fixed_width<int>(right_children_col, tree_row, cudf::data_type{cudf::type_id::INT32}, rights)) return false;
    if (!read_list_row_fixed_width<float>(split_conditions_col, tree_row, cudf::data_type{cudf::type_id::FLOAT32}, thresholds)) return false;
    if (!read_list_row_fixed_width<int>(split_indices_col, tree_row, cudf::data_type{cudf::type_id::INT32}, features)) return false;
    if (!read_list_row_fixed_width<float>(base_weights_col, tree_row, cudf::data_type{cudf::type_id::FLOAT32}, leaf_values)) return false;
    if (!read_list_row_fixed_width<int8_t>(default_left_col, tree_row, cudf::data_type{cudf::type_id::INT8}, defaults)) return false;

    if (lefts.size() != rights.size() ||
        lefts.size() != thresholds.size() ||
        lefts.size() != features.size() ||
        lefts.size() != leaf_values.size() ||
        lefts.size() != defaults.size())
{
      std::cerr << "[xgb] inconsistent tree node array sizes at tree " << t << "\n";
      return false;
    }

    int base_index = static_cast<int>(model.nodes.size());
    model.entry_nodes.push_back(base_index);

    for (size_t i = 0; i < lefts.size(); ++i)
{
      Node node{};
      int l = lefts[i];
      int r = rights[i];

      if (l == -1 && r == -1)
{
        node.feature = -1;
        node.threshold = 0.0f;
        node.left = -1;
        node.right = -1;
        node.leaf = leaf_values[i];
        node.is_leaf = 1;
        node.default_left = 0;
      } else
{
        node.feature = features[i];
        node.threshold = thresholds[i];
        node.left = base_index + l;
        node.right = base_index + r;
        node.leaf = 0.0f;
        node.is_leaf = 0;
        node.default_left = static_cast<int8_t>(defaults[i] != 0);
      }
      model.nodes.push_back(node);
    }
  }

  return true;
}

}  // namespace echter::xgb::detail

namespace echter::xgb::detail
{

bool upload_model(const HostModel& host, DeviceModel& device)
{
    device.release();

    device.num_nodes = static_cast<int>(host.nodes.size());
    device.num_trees = static_cast<int>(host.entry_nodes.size());
    device.num_features = host.num_features;
    device.base_score = host.base_score;

    if (device.num_nodes == 0 || device.num_trees == 0)
    {
        device.release();
        return false;
    }

    try
    {
        if (cudaMalloc(
                reinterpret_cast<void**>(&device.nodes),
                host.nodes.size() * sizeof(Node)) != cudaSuccess)
        {
            device.release();
            return false;
        }

        if (cudaMalloc(
                reinterpret_cast<void**>(&device.entry_nodes),
                host.entry_nodes.size() * sizeof(int)) != cudaSuccess)
        {
            device.release();
            return false;
        }

        if (cudaMemcpy(
                device.nodes,
                host.nodes.data(),
                host.nodes.size() * sizeof(Node),
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            device.release();
            return false;
        }

        if (cudaMemcpy(
                device.entry_nodes,
                host.entry_nodes.data(),
                host.entry_nodes.size() * sizeof(int),
                cudaMemcpyHostToDevice) != cudaSuccess)
        {
            device.release();
            return false;
        }
    }
    catch (...)
    {
        device.release();
        return false;
    }

    return true;
}

}
