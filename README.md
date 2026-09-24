# Echter_XGBoost

CUDA XGBoost inference library using C++20 named modules. The current model implementation supports regression. The module and directory layout is prepared for later classification support.

## Build

Requirements:

- CMake 3.28 or newer
- CUDA toolkit and `nvcc`
- cuDF
- C++20 module support
- Ninja is recommended

```bash
cmake -S . -B build -G Ninja
cmake --build build
cmake --install build --strip
```

The build creates the library, the CSV application, the test executable, and four examples. `CMAKE_BUILD_TYPE` defaults to `Release`. The install step copies the executables to `bin/` in the source tree (override with `--prefix`); they keep the RPATH to the CUDA and cuDF libraries they were linked against, so they run on the build machine without setting `LD_LIBRARY_PATH`.

## Supported models

`Regression::load_model()` reads XGBoost JSON models (`Booster.save_model("model.json")`) with:

- the `gbtree` booster and a single target,
- numerical splits (categorical splits are rejected),
- an objective without an output transformation: `reg:squarederror`, `reg:squaredlogerror`, `reg:pseudohubererror`, `reg:absoluteerror`, or `reg:quantileerror` with one quantile.

Other objectives, such as `reg:logistic`, `count:poisson`, or `reg:gamma`, are rejected with an error instead of returning untransformed margins. The loader also validates every tree before it is uploaded: split features must exist, and child links must stay inside the tree and form a tree. Leaf values are read from `split_conditions`, as XGBoost does.

Predictions are bit-identical to XGBoost's GPU predictor: like XGBoost, the kernel sums the leaf values in tree order and adds the sum to the base score at the end.

Missing values are `NaN`. A missing feature follows the node's default direction, like in XGBoost.

## Public modules

The main user entry point is:

```cpp
import echter.xgb;
```

It exports the `XGBoost` facade, the regression API, the common data types, and the IO API.

The model API is used directly by applications. Model loading does not require the caller to use the IO API manually:

```cpp
import echter.xgb;

echter::xgb::XGBoost api;
auto& regression = api.regression();
regression.load_model("model.json");
```

The common data module is also available directly:

```cpp
import echter.xgb.data;
```

The optional public file module is:

```cpp
import echter.xgb.io;
```

The IO module is not the main model API. It is an optional public utility for applications that want direct CSV or JSON file access.

## IO API

The IO implementation uses cuDF and device memory. It does not use `fstream` and the public IO module does not include CUDA or cuDF headers.

```cpp
import echter.xgb;

const auto table = echter::xgb::io::read_csv("input.csv");
const auto features =
    echter::xgb::io::select_columns(table, 0);
const auto prediction = regression.predict(features);

echter::xgb::io::write_csv(
    "output.csv",
    std::vector<echter::xgb::DeviceColumnarView>{
        {table.view().data, table.rows(), 1},
        prediction.values.view()},
    {"id", "prediction"});
```

Available operations:

- `io::read_json(path)` reads a JSON file through the cuDF datasource layer.
- `io::read_csv(path, has_header)` reads numeric CSV columns into GPU memory. Empty fields become `NaN`, that is, missing values.
- `io::select_columns(table, excluded_column)` creates a device-only column selection.
- `io::select_columns(table, excluded_columns)` removes multiple columns in one device-only selection.
- `io::write_csv(path, table, names)` writes every column of a device table through cuDF.
- `io::write_csv(path, columns, names)` writes multiple device columns through cuDF.

Without `names`, no header is written; otherwise there must be one name per column. Failures are reported with exceptions whose message names the file and the cause.

## Data and prediction paths

The primitive input layout is column-major:

```text
data[feature * rows + row]
```

CPU input uses `HostColumnarView`. The host overload uploads the data to the GPU and copies only the final prediction vector back to the CPU:

```cpp
const auto result = regression.predict(
    echter::xgb::HostColumnarView{host_data, rows, features});
```

GPU input uses `DeviceColumnarView` or `DeviceColumnarData`. It returns `DevicePrediction`; the result remains on the GPU:

```cpp
const auto result = regression.predict(device_data);
```

The cuDF adapter also keeps input and prediction data on the GPU. CPU copies are performed only when an application explicitly requests them, for example to print sample values.

CUDA-specific implementation files are kept in `.cu` and `.cuh` files. The public modules and module implementation files do not depend on CUDA or cuDF headers.

## Application

`apps/predict.cpp` is the normal CSV prediction program. It imports only the public module and does not import CUDA or cuDF directly.

Usage:

```bash
./bin/echter_xgb_predict <model.json> <input.csv> [key-column]
```

Arguments:

- `model.json`: XGBoost JSON model path.
- `input.csv`: numeric CSV input path. The file may contain an identifier column and feature columns.
- `key-column`: optional zero-based identifier column index. The default is `0`.

Multiple columns can be excluded from prediction with the `--target-columns` and
`--ignore-columns` options. Both options remove their listed columns from the
feature matrix; the separate names make it clear which columns are targets and
which are simply not model inputs:

```bash
./bin/echter_xgb_predict model.json input.csv \
    --key-column 0 \
    --target-columns 1,2 \
    --ignore-columns 5,7
```

Indexes are zero-based and refer to the original CSV. The options may be
repeated, and duplicate indexes are ignored. The key column is written as
`id`; every requested target is also written as `target_<index>`, followed by
`prediction`. Ignored columns are not written. The remaining column count must
match the model feature count.

The app loads the model through `Regression::load_model()`, reads the CSV through `io::read_csv()`, removes the key, target, and ignored columns with device-to-device copying, and predicts without copying the input or prediction to the CPU. The result is written through `io::write_csv()` to:

```text
outputs/<input-stem>_predictions.csv
```

The output contains the key, the requested targets, and the prediction:

```text
id,target_<index>...,prediction
```

The application prints model path, input path, output path, row count, column count, feature count, tree count, and model-load, CSV-read, prediction, CSV-write, and total timings.

## Examples

The CPU, CUDA, and cuDF examples take the model path and an optional row count. The default row count is `100000`, so a large input can be tested without a CSV file.

### CPU API

```bash
./bin/echter_xgb_cpu_prediction /path/to/model.json 100000
```

This example generates a CPU host array, calls the host prediction overload, prints model and timing statistics, and prints at most the first ten predictions.

### CSV API

```bash
./bin/echter_xgb_csv_prediction /path/to/model.json /path/to/input.csv \
    --target-columns 1 --ignore-columns 5,7
```

This is a small CSV-focused example of the public API. It uses only
`import echter.xgb`, reads the input directly through the IO module, excludes
the key, target, and ignored columns, and writes the key, targets, and
predictions to the `outputs/` directory. It does not import CUDA or cuDF
directly, and it prints only the row count, tree count, and output path. The
full application in `apps/predict.cpp` provides the same workflow with
equivalent command-line options and detailed timing.

### Direct CUDA API

```bash
./bin/echter_xgb_cuda_prediction /path/to/model.json 100000
```

This example generates random input, allocates a device buffer through the model API, uploads it with CUDA, runs the device prediction overload, and prints at most the first ten values after explicitly copying only those sample values to the CPU.

### cuDF API

```bash
./bin/echter_xgb_cudf_prediction /path/to/model.json 100000
```

This example creates random cuDF device columns, calls the public cuDF adapter, and prints model and timing statistics, including finite-value count, minimum, maximum, mean, and standard deviation. The first ten values are also copied to the CPU for display.

The CPU, CUDA, and cuDF examples do not read or write CSV files. They generate input based on `regression.num_features()` and the requested row count.

## Tests

```bash
./bin/echter_xgb_test                                # built-in tests
./bin/echter_xgb_test /path/to/model.json [rows]     # built-in tests and a real model
ctest --test-dir build                               # built-in tests through CTest
```

Without arguments the test writes a small two-tree XGBoost model and CSV files to the temporary directory and checks exact predictions, including missing values and CSV fields, the host and device paths, CSV round trips, and the rejection of invalid models (unsupported objective, bad child links, cycles, unknown split features). With a model path it also predicts random rows with that model and checks that the host and device results agree and are finite. Configure with `-DECHTER_XGB_TEST_MODEL=/path/to/model.json` to add the model run to CTest.

## Reference script

`scripts/predict.py` predicts the same CSV with the XGBoost Python package on the GPU, for comparing results and timings with `echter_xgb_predict`. It needs `xgboost` and `pandas`, and selects the feature columns by the model's feature names:

```bash
python scripts/predict.py model.json input.csv --target-column incident_proton_energy
```

`--key-column` (default: the first CSV column) is written as `id` and `--target-column` as `target`. The output goes to `outputs/<input-stem>_xgboost_predictions.csv` unless `--output` is given.

## Source layout

```text
src/
├── core/
│   ├── cuda_utils.cuh
│   ├── cudf_convert.cuh
│   ├── model_backend.hpp
│   └── model_backend.cu
├── io/
│   ├── echter.xgb.io.cppm
│   ├── echter.xgb.io.cpp
│   ├── io_backend.hpp
│   └── io_backend.cu
├── modules/
│   ├── echter.cppm
│   ├── echter.xgb.cppm
│   ├── echter.xgb.cpp
│   ├── echter.xgb.data.cppm
│   ├── echter.xgb.data.cpp
│   ├── echter.xgb.reg.cppm
│   └── echter.xgb.reg.cpp
└── regression/
    ├── regression_backend.hpp
    ├── regression_cudf_adapter.cpp
    ├── regression_cudf_adapter.hpp
    ├── regression_kernels.cu
    ├── regression_kernels.cuh
    ├── regression_model.cuh
    ├── regression_model_loader.cpp
    └── regression_runtime.cu
```

A future `classification` implementation can add a separate module and directory while reusing `echter.xgb.data`, the generic device backend, and the public IO module.
