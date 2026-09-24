#include "regression_internal.cuh"
#include "regression_kernels.cuh"
#include "regression_backend.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace echter::xgb::detail
{

namespace
{

void check_cuda(cudaError_t error, const std::string& operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(operation + ": " + cudaGetErrorString(error));
    }
}

}

}

namespace echter::xgb::regression_backend
{

void* create_model()
{
    return new detail::DeviceModel{};
}

void destroy_model(void* model) noexcept
{
    delete static_cast<detail::DeviceModel*>(model);
}

bool load_model(void* model, const std::string& path)
{
    if (model == nullptr)
    {
        return false;
    }

    detail::HostModel host;
    return detail::load_model_with_cudf(path, host)
        && detail::upload_model(host, *static_cast<detail::DeviceModel*>(model));
}

int model_features(const void* model) noexcept
{
    return model == nullptr
        ? 0
        : static_cast<const detail::DeviceModel*>(model)->num_features;
}

int model_trees(const void* model) noexcept
{
    return model == nullptr
        ? 0
        : static_cast<const detail::DeviceModel*>(model)->num_trees;
}

bool predict_device(
    model_backend::DeviceView device,
    const void* model,
    void* output)
{
    if (model == nullptr || output == nullptr)
    {
        return false;
    }

    detail::DeviceColumnarBuffer input{
        const_cast<float*>(device.data), device.rows, device.features};
    return detail::run_regression_device(
        input,
        *static_cast<const detail::DeviceModel*>(model),
        *static_cast<detail::DeviceColumnarBuffer*>(output));
}

}

namespace echter::xgb::detail
{

void DeviceColumnarBuffer::release() noexcept
{
    if (data != nullptr)
    {
        cudaFree(data);
        data = nullptr;
    }

    rows = 0;
    features = 0;
}

void DeviceModel::release() noexcept
{
    if (nodes != nullptr)
    {
        cudaFree(nodes);
        nodes = nullptr;
    }

    if (entry_nodes != nullptr)
    {
        cudaFree(entry_nodes);
        entry_nodes = nullptr;
    }

    num_nodes = 0;
    num_trees = 0;
    num_features = 0;
    base_score = 0.0f;
}

bool allocate_columnar(
    std::size_t rows,
    std::size_t features,
    DeviceColumnarBuffer& out)
{
    out.release();
    out.rows = rows;
    out.features = features;

    if (rows == 0 || features == 0)
    {
        return true;
    }

    try
    {
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&out.data),
                rows * features * sizeof(float)),
            "cudaMalloc(columnar input)");
    }
    catch (...)
    {
        out.release();
        return false;
    }

    return true;
}

bool upload_columnar(
    const float* host,
    std::size_t rows,
    std::size_t features,
    DeviceColumnarBuffer& out)
{
    if (!allocate_columnar(rows, features, out))
    {
        return false;
    }

    if (rows == 0 || features == 0)
    {
        return true;
    }

    try
    {
        check_cuda(
            cudaMemcpy(
                out.data,
                host,
                rows * features * sizeof(float),
                cudaMemcpyHostToDevice),
            "cudaMemcpy(columnar input H2D)");
    }
    catch (...)
    {
        out.release();
        return false;
    }

    return true;
}

bool run_regression_device(
    const DeviceColumnarBuffer& input,
    const DeviceModel& model,
    DeviceColumnarBuffer& output)
{
    if (input.rows == 0)
    {
        return true;
    }

    try
    {
        if (output.data == nullptr
            || output.rows != input.rows
            || output.features != 1)
        {
            return false;
        }

        launch_regression_kernel(
            input.data,
            output.data,
            input.rows,
            input.features,
            model);

        check_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize(prediction)");

        return true;
    }
    catch (...)
    {
        return false;
    }
}

}

namespace echter::xgb::detail
{

DeviceModel::~DeviceModel()
{
    release();
}

DeviceModel::DeviceModel(DeviceModel&& other) noexcept
    : nodes(other.nodes),
      entry_nodes(other.entry_nodes),
      num_nodes(other.num_nodes),
      num_trees(other.num_trees),
      num_features(other.num_features),
      base_score(other.base_score)
{
    other.nodes = nullptr;
    other.entry_nodes = nullptr;
    other.num_nodes = 0;
    other.num_trees = 0;
    other.num_features = 0;
    other.base_score = 0.0f;
}

DeviceModel& DeviceModel::operator=(DeviceModel&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }

    release();

    nodes = other.nodes;
    entry_nodes = other.entry_nodes;
    num_nodes = other.num_nodes;
    num_trees = other.num_trees;
    num_features = other.num_features;
    base_score = other.base_score;

    other.nodes = nullptr;
    other.entry_nodes = nullptr;
    other.num_nodes = 0;
    other.num_trees = 0;
    other.num_features = 0;
    other.base_score = 0.0f;

    return *this;
}

}
