# Echter XGBoost

![C++20 modules](https://img.shields.io/badge/C%2B%2B20-modules-blue)
![CUDA 13](https://img.shields.io/badge/CUDA-13-76B900)
![cuDF 26.02](https://img.shields.io/badge/cuDF-26.02-7400B8)
![CMake 3.28+](https://img.shields.io/badge/CMake-3.28%2B-064F8C)

GPU inference for XGBoost regression models in C++20. Echter XGBoost loads a model saved by XGBoost, predicts on the GPU with a CUDA kernel, and reads and writes CSV files on the GPU through cuDF.

- **Same results as XGBoost:** predictions are bit-identical to XGBoost's GPU predictor.
- **Fast:** 75 million rows per second on an H200, 2.3–2.7× the throughput of XGBoost's C API on the same GPU, 3.7–6.2× with host input, and 14× faster model loading. See [Performance](#performance).
- **GPU end to end:** CSV input, column selection, prediction, and CSV output without a round trip through host memory.
- **Safe to use:** models are validated before they are uploaded, device memory is owned by RAII types, and host pointers passed as device data are rejected before a kernel runs.
- **Modern C++:** C++20 named modules; the public modules do not expose CUDA or cuDF headers.

## Contents

- [Performance](#performance)
- [Requirements](#requirements)
- [Build and install](#build-and-install)
- [Command-line prediction](#command-line-prediction)
- [Library usage](#library-usage)
- [Supported models](#supported-models)
- [Examples](#examples)
- [Tests](#tests)
- [Comparing with XGBoost](#comparing-with-xgboost)
- [Project layout](#project-layout)

## Performance

With the benchmark model below, Echter XGBoost predicts **about 75 million rows per second** on an H200 when the input is already in GPU memory, and about 40 million rows per second from host memory. XGBoost reaches 26–33 million and 6–11 million rows per second in the same situations.

### Setup

- **Hardware:** NVIDIA H200 GPU, Intel Xeon Platinum 8480C CPU
- **Software:** CUDA 13.1, cuDF 26.02, XGBoost 3.4.1 with `device="cuda"`
- **XGBoost interfaces:** the Python package (`Booster.inplace_predict` on a NumPy array) and the C API called from C++ (`XGBoosterPredictFromDense` for host input, `XGBoosterPredictFromCudaArray` for GPU input)
- **Data:** 489,046 real rows with 19 features, repeated 10× and 20× for the larger batches
- **Method:** median of 9 calls per run (5 for host input to XGBoost), median over repeated runs

All implementations and input paths produce byte-identical predictions; this was checked on all 15.2 M benchmark rows.

### Benchmark model

| Property | Value |
|---|---|
| Trees | 500 (`reg:squarederror`, 19 features) |
| Nodes | 6,715,914 in total: 3,357,707 splits and 3,358,207 leaves, about 13,400 per tree |
| Depth | Average leaf depth 14.7, maximum depth 16 |
| Model file (JSON) | 387 MB |
| Trees in GPU memory | 107 MB, 16 bytes per node |

### Comparison with XGBoost

Rows per second, and how many times faster Echter XGBoost is.

**Input already in GPU memory**

| Rows | XGBoost C API | Echter | Speedup |
|---:|---:|---:|---:|
| 489,046 | 25.9 M rows/s | 69.8 M rows/s | **2.7×** |
| 4,890,460 | 32.1 M rows/s | 74.9 M rows/s | **2.3×** |
| 9,780,920 | 32.7 M rows/s | 75.8 M rows/s | **2.3×** |

**Input in host memory** (every call copies the input to the GPU and the predictions back)

| Rows | XGBoost Python | XGBoost C API | Echter | Speedup vs Python | Speedup vs C API |
|---:|---:|---:|---:|---:|---:|
| 489,046 | 6.8 M rows/s | 6.3 M rows/s | 39.4 M rows/s | **5.8×** | **6.2×** |
| 4,890,460 | 10.9 M rows/s | 9.7 M rows/s | 43.6 M rows/s | **4.0×** | **4.5×** |
| 9,780,920 | 10.8 M rows/s | 11.1 M rows/s | 40.4 M rows/s | **3.8×** | **3.7×** |

With host input and a GPU booster, XGBoost first builds a DMatrix from the data, as its own performance warning says. The Python package needs CuPy or cuDF for GPU input, so the GPU-input comparison uses the C API.

**Model loading** (the 387 MB JSON file)

| XGBoost Python | XGBoost C API | Echter | Speedup |
|---:|---:|---:|---:|
| 11.9 s | 12.3 s | 0.86 s, including CUDA initialization | **14×** |

### Input paths of Echter XGBoost

The same model and data through the four ways a table can reach the model, in rows per second. Each path has an example program in `bin/examples/`.

| Input | API | Example | 489,046 rows | 4,890,460 rows | 9,780,920 rows |
|---|---|---|---:|---:|---:|
| CUDA device buffer | `predict(DeviceColumnarData)` | `echter_xgb_cuda_prediction` | 69.8 M/s | 74.9 M/s | 75.8 M/s |
| cuDF table | `predict(regression, cudf::table_view)` | `echter_xgb_cudf_prediction` | 65.3 M/s | 73.4 M/s | 74.8 M/s |
| Host array | `predict(HostColumnarView)` | `echter_xgb_cpu_prediction` | 39.4 M/s | 43.6 M/s | 40.4 M/s |
| CSV file | `io::read_csv`, `io::select_columns`, `predict` | `echter_xgb_csv_prediction` | 9.6 M/s | 13.7 M/s | 14.2 M/s |

The cuDF path includes packing the table's columns into one column-major buffer. The host path includes both copies between host and GPU. The CSV path includes reading and parsing the file on the GPU; the 9,780,920-row file is 3.0 GB.

### CSV to CSV

The full pipeline on the 489,046-row, 21-column CSV file: [`scripts/predict.py`](scripts/predict.py) (pandas and XGBoost) against `echter_xgb_predict`, median of three runs.

| Step | XGBoost (Python) | Echter | Speedup |
|---|---:|---:|---:|
| Model load | 12,559 ms | 817 ms | 15× |
| CSV read | 1,127 ms | 63 ms | 18× |
| Prediction | 357 ms (DMatrix 68 + predict 289) | 7 ms | 51× |
| CSV write | 511 ms | 31 ms | 16× |
| **Total** | **14.7 s** | **0.94 s** | **16×** |
| Rows per second, whole pipeline | 33 k | 518 k | |

## Requirements

- Linux with an NVIDIA GPU of compute capability 7.5 or newer
- CUDA Toolkit (tested with 13.1)
- libcudf, the cuDF C++ library (tested with 26.02, for example from the RAPIDS conda packages)
- CMake 3.28 or newer and Ninja 1.11 or newer (C++20 modules do not work with the Makefile generators)
- A C++20 compiler with module support (tested with GCC 15.2 and Clang 21.1)

## Build and install

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
cmake --install build --strip
```

If CMake does not find cuDF, add its installation prefix, for example `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX`.

`cmake --install` puts the command-line application into `bin/`, the test into `bin/test/`, and the examples into `bin/examples/` inside the source tree; use `--prefix` to install somewhere else. Installed executables keep the RPATH to the CUDA and cuDF libraries they were linked against.

The executables in this repository's `bin/` are built on the benchmark machine for all GPU architectures listed below. They load CUDA and cuDF from that machine's paths, so on other machines rebuild them or point `LD_LIBRARY_PATH` to matching libraries.

| CMake option | Default | Purpose |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `all-major` (sm_75 to sm_120, plus PTX) | Set to `native` to build only for the local GPU; the `CUDAARCHS` environment variable also works |
| `CMAKE_BUILD_TYPE` | `Release` | |
| `ECHTER_XGB_BUILD_PREDICT` | `ON` | The `echter_xgb_predict` application |
| `ECHTER_XGB_BUILD_TESTS` | `ON` | The test executable and CTest registration |
| `ECHTER_XGB_BUILD_EXAMPLES` | `ON` | The four examples |
| `ECHTER_XGB_TEST_MODEL` | empty | A model JSON for an extra CTest run with a real model |

The three `BUILD` options default to `OFF` when the project is included from another CMake project.

## Command-line prediction

```bash
./bin/echter_xgb_predict model.json input.csv --target-columns 1
```

| Argument | Meaning |
|---|---|
| `model.json` | XGBoost JSON model |
| `input.csv` | Numeric CSV with a header row |
| `[key-column]`, `--key-column INDEX` | Identifier column, written as `id` (default `0`) |
| `--target-columns INDEX[,INDEX...]` | Columns that are not model inputs but are copied to the output as `target_<index>` |
| `--ignore-columns INDEX[,INDEX...]` | Columns that are neither model inputs nor written |

Indexes are zero-based, options can be repeated, and duplicate indexes are ignored. All remaining columns are features, and their number must match the model. The result is written to `outputs/<input-stem>_predictions.csv` with the columns `id,target_<index>...,prediction`. The program prints the row and column counts and the time of every step.

## Library usage

```cpp
#include <vector>

import echter.xgb;

int main()
{
    echter::xgb::Regression regression;
    regression.load_model("model.json");

    // Column-major input, data[feature * rows + row]; NaN marks a missing value.
    const std::size_t rows = 2;
    const std::size_t features = regression.num_features();
    std::vector<float> data(rows * features, 0.0f);

    const auto result = regression.predict(
        echter::xgb::HostColumnarView{data.data(), rows, features});
    // result.values holds one prediction per row.
}
```

Data can stay on the GPU from input to output. `predict()` returns a `DevicePrediction` for device input:

```cpp
namespace io = echter::xgb::io;

const auto table = io::read_csv("input.csv");             // numeric columns on the GPU
const auto features = io::select_columns(table, {0, 1});  // drop the id and target columns
const auto prediction = regression.predict(features);     // result stays on the GPU

io::write_csv(
    "predictions.csv",
    std::vector<echter::xgb::DeviceColumnarView>{
        {table.view().data, table.rows(), 1},             // column 0 of the table
        prediction.values.view()},
    {"id", "prediction"});
```

| Module | Contents |
|---|---|
| `echter.xgb` | All modules below, and the `XGBoost` facade with `regression()` |
| `echter.xgb.reg` | `Regression`: `load_model`, `predict` for host and device input, `upload`, `allocate_device`, `num_features`, `num_trees` |
| `echter.xgb.data` | `HostColumnarView`, `DeviceColumnarView`, `DeviceColumnarData` (owns device memory; `view()`, `mutable_data()`), `Prediction`, `DevicePrediction` |
| `echter.xgb.io` | `read_csv`, `select_columns`, `write_csv`, `read_json` |

Notes:

- `io::read_csv` converts all columns to `float`; empty fields become `NaN`. `io::write_csv` writes a header only when names are given, one per column.
- A `DeviceColumnarView` must point to device memory. Host pointers are rejected with an exception.
- Errors are reported with exceptions whose messages name the file and the cause.
- Moved-from `Regression` and `DeviceColumnarData` objects are empty; a moved-from model throws "not loaded" when used.

To use the library from another CMake project, add it as a subdirectory. The C++20 requirement propagates to the consumer:

```cmake
find_package(CUDAToolkit REQUIRED)
find_package(cudf CONFIG REQUIRED)
add_subdirectory(Echter_XGBoost)
target_link_libraries(my_app PRIVATE Echter_XGBoost::Echter_XGBoost)
```

## Supported models

`Regression::load_model()` reads models saved with `Booster.save_model("model.json")` that use:

- the `gbtree` booster with a single target,
- numerical splits,
- an objective without an output transformation: `reg:squarederror`, `reg:squaredlogerror`, `reg:pseudohubererror`, `reg:absoluteerror`, or `reg:quantileerror` with one quantile.

Other models are rejected with an error instead of producing wrong numbers. This includes objectives such as `reg:logistic`, `count:poisson`, or `reg:gamma`, categorical splits, multi-target models, and classification models. Every tree is validated before it is uploaded: split features must exist, and child links must stay inside the tree without cycles.

Missing values (`NaN`) follow each split's default direction. Leaf values are summed in tree order and added to the base score at the end, exactly as XGBoost's GPU predictor does, which makes the results bit-identical.

## Examples

The examples are installed into `bin/examples/`.

| Executable | Shows |
|---|---|
| `echter_xgb_cpu_prediction <model.json> [rows]` | Host input and host output |
| `echter_xgb_cuda_prediction <model.json> [rows]` | Filling a device buffer with CUDA and predicting on the GPU |
| `echter_xgb_cudf_prediction <model.json> [rows]` | Predicting from a `cudf::table_view`, with output statistics |
| `echter_xgb_csv_prediction <model.json> <input.csv> [options]` | A compact version of the command-line application |

The first three generate random input with `rows` rows (default 100,000) and print the first ten predictions.

## Tests

```bash
./bin/test/echter_xgb_test                           # built-in tests
./bin/test/echter_xgb_test model.json [rows]         # built-in tests and a real model
```

The built-in tests generate a small XGBoost model and CSV files in the temporary directory and check:

- exact predictions, including missing values, empty CSV fields, and the summation order,
- the host and device prediction paths and CSV round trips,
- the rejection of invalid models (unsupported objective, bad child links, cycles, unknown split features),
- memory safety (moved-from objects and host pointers passed as device data).

With a model file, the test also checks that host and device predictions of random rows agree and are finite. Configure with `-DECHTER_XGB_TEST_MODEL=model.json` to add that run to `ctest`.

## Comparing with XGBoost

[`scripts/predict.py`](scripts/predict.py) runs the same CSV pipeline with pandas and the XGBoost Python package on the GPU. Its output can be compared with the output of `echter_xgb_predict`, and it prints the same timing breakdown. It needs `xgboost` and `pandas`:

```bash
python scripts/predict.py model.json input.csv --target-column incident_proton_energy
```

The feature columns are selected by the model's feature names. `--key-column` (default: the first column) is written as `id`, and `--target-column` is written as `target`. The output goes to `outputs/<input-stem>_xgboost_predictions.csv` unless `--output` is given.

## Project layout

```text
apps/          echter_xgb_predict command-line application
bin/           prebuilt application; bin/test/ holds the test, bin/examples/ the examples
examples/      CPU, CUDA, cuDF, and CSV examples
scripts/       XGBoost reference script
src/
├── core/        device memory (RAII buffers), CUDA error checks, cuDF conversion
├── io/          echter.xgb.io module and its cuDF CSV backend
├── modules/     echter.xgb, echter.xgb.data, echter.xgb.reg modules
└── regression/  model loader, CUDA kernel, runtime, cuDF adapter
tests/         self-contained tests
```

Source files follow one convention:

| Extension | Contents |
|---|---|
| `.hpp` | API headers; they never include CUDA or cuDF headers, so module units can use them |
| `.cuh` | Headers that include CUDA (or cuDF) and declare kernels |
| `.cu` | Kernel implementations, the only files compiled by `nvcc` |
| `.cpp` | Host code, including the code that calls the CUDA runtime or cuDF |

The prediction kernel, for example, is split into `regression_kernels.hpp` (the launch function), `regression_kernels.cuh` (CUDA includes and the kernel declaration), and `regression_kernels.cu` (the kernel). The module structure leaves room for classification support next to `regression/`.
