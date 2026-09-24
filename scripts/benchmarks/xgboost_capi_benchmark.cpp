#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using BoosterHandle = void*;
using bst_ulong = std::uint64_t;
using clock_type = std::chrono::steady_clock;

class XGBoostApi
{
public:
    explicit XGBoostApi(const std::string& library_path)
        : library_(dlopen(library_path.c_str(), RTLD_NOW))
    {
        if (library_ == nullptr)
        {
            throw std::runtime_error(dlerror());
        }
        load(get_last_error, "XGBGetLastError");
        load(create, "XGBoosterCreate");
        load(free_booster, "XGBoosterFree");
        load(load_model, "XGBoosterLoadModel");
        load(set_param, "XGBoosterSetParam");
        load(predict_from_dense, "XGBoosterPredictFromDense");
        load(predict_from_cuda_array, "XGBoosterPredictFromCudaArray");
    }

    void check(int status) const
    {
        if (status != 0)
        {
            throw std::runtime_error(get_last_error());
        }
    }

    const char* (*get_last_error)() = nullptr;
    int (*create)(void* const*, bst_ulong, BoosterHandle*) = nullptr;
    int (*free_booster)(BoosterHandle) = nullptr;
    int (*load_model)(BoosterHandle, const char*) = nullptr;
    int (*set_param)(BoosterHandle, const char*, const char*) = nullptr;
    int (*predict_from_dense)(BoosterHandle, const char*, const char*, void*,
                              const bst_ulong**, bst_ulong*, const float**) = nullptr;
    int (*predict_from_cuda_array)(BoosterHandle, const char*, const char*, void*,
                                   const bst_ulong**, bst_ulong*, const float**) = nullptr;

private:
    template <typename Function>
    void load(Function& function, const char* name)
    {
        function = reinterpret_cast<Function>(dlsym(library_, name));
        if (function == nullptr)
        {
            throw std::runtime_error(std::string("missing symbol ") + name);
        }
    }

    void* library_;
};

double median_ms(int repetitions, const std::function<void()>& run)
{
    run();
    std::vector<double> times;
    for (int repetition = 0; repetition < repetitions; ++repetition)
    {
        const auto start = clock_type::now();
        run();
        times.push_back(std::chrono::duration<double, std::milli>(clock_type::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

std::string array_interface(const void* data, std::size_t rows, std::size_t features, bool host)
{
    return "{\"data\": [" + std::to_string(reinterpret_cast<std::uintptr_t>(data)) + ", "
        + (host ? "true" : "false") + "], \"shape\": [" + std::to_string(rows) + ", "
        + std::to_string(features) + "], \"typestr\": \"<f4\", \"version\": 3}";
}

}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: xgboost_capi_benchmark <libxgboost.so> <model.json> <output-dir>\n";
        return 2;
    }

    try
    {
        const XGBoostApi xgboost(argv[1]);
        const std::filesystem::path output_dir = argv[3];

        std::ifstream in(output_dir / "features.bin", std::ios::binary);
        std::uint64_t shape[2]{};
        in.read(reinterpret_cast<char*>(shape), sizeof(shape));
        const std::size_t base_rows = shape[0];
        const std::size_t features = shape[1];
        std::vector<float> base(base_rows * features);
        in.read(reinterpret_cast<char*>(base.data()), base.size() * sizeof(float));

        BoosterHandle booster = nullptr;
        const auto load_start = clock_type::now();
        xgboost.check(xgboost.create(nullptr, 0, &booster));
        xgboost.check(xgboost.load_model(booster, argv[2]));
        xgboost.check(xgboost.set_param(booster, "device", "cuda"));
        std::cout << std::fixed << std::setprecision(2) << "model load "
                  << std::chrono::duration<double, std::milli>(clock_type::now() - load_start).count()
                  << " ms\n";

        const char* config = "{\"type\": 0, \"training\": false, \"iteration_begin\": 0, "
                             "\"iteration_end\": 0, \"missing\": NaN, \"strict_shape\": false, "
                             "\"cache_id\": 0}";
        const bst_ulong* out_shape = nullptr;
        bst_ulong out_dim = 0;
        const float* out_result = nullptr;

        for (std::size_t factor : {1, 10, 20})
        {
            const std::size_t rows = base_rows * factor;
            std::vector<float> host(rows * features);
            for (std::size_t row = 0; row < rows; ++row)
            {
                for (std::size_t feature = 0; feature < features; ++feature)
                {
                    host[row * features + feature] = base[feature * base_rows + row % base_rows];
                }
            }

            const auto host_interface = array_interface(host.data(), rows, features, true);
            const double host_ms = median_ms(5, [&] {
                xgboost.check(xgboost.predict_from_dense(booster, host_interface.c_str(), config,
                                                         nullptr, &out_shape, &out_dim, &out_result));
            });

            float* device = nullptr;
            cudaMalloc(&device, host.size() * sizeof(float));
            cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
            const auto device_interface = array_interface(device, rows, features, false);
            const double device_ms = median_ms(9, [&] {
                xgboost.check(xgboost.predict_from_cuda_array(booster, device_interface.c_str(), config,
                                                              nullptr, &out_shape, &out_dim, &out_result));
                cudaDeviceSynchronize();
            });
            std::vector<float> prediction(rows);
            cudaMemcpy(prediction.data(), out_result, rows * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(device);

            const auto rate = [rows](double ms) { return rows / ms / 1000.0; };
            std::cout << "rows " << std::setw(8) << rows
                      << " | device " << device_ms << " ms " << rate(device_ms) << " M/s"
                      << " | host " << host_ms << " ms " << rate(host_ms) << " M/s\n";

            std::ofstream(output_dir / ("xgboost_capi_x" + std::to_string(factor) + ".bin"), std::ios::binary)
                .write(reinterpret_cast<const char*>(prediction.data()), prediction.size() * sizeof(float));
        }
        xgboost.check(xgboost.free_booster(booster));
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
