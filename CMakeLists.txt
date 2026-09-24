cmake_minimum_required(VERSION 3.28)

project(Echter_XGBoost VERSION 0.3.0 LANGUAGES CXX CUDA)

option(ECHTER_XGB_BUILD_TESTS "Build test executable" ON)
option(ECHTER_XGB_BUILD_PREDICT "Build CSV prediction executable" ON)
option(ECHTER_XGB_BUILD_EXAMPLES "Build examples" ON)
set(ECHTER_XGB_TEST_MODEL "" CACHE FILEPATH "Model JSON used by CTest runtime test")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(cudf CONFIG REQUIRED)

add_library(Echter_XGBoost STATIC
    src/regression/regression_model_loader.cpp
    src/regression/regression_runtime.cu
    src/regression/regression_cudf_adapter.cpp
    src/regression/regression_backend.hpp
    src/core/model_backend.cu
    src/modules/echter.xgb.reg.cpp
    src/modules/echter.xgb.cpp
    src/modules/echter.xgb.data.cpp
    src/io/echter.xgb.io.cpp
    src/io/io_backend.cu
    src/io/io_backend.hpp
    src/regression/regression_kernels.cu
    src/regression/regression_model.cuh
    src/regression/regression_internal.cuh
    src/regression/regression_kernels.cuh
    src/regression/regression_cudf_adapter.hpp
    src/core/model_backend.hpp
)

target_sources(Echter_XGBoost
    PUBLIC
        FILE_SET CXX_MODULES TYPE CXX_MODULES FILES
            src/modules/echter.cppm
            src/modules/echter.xgb.data.cppm
            src/io/echter.xgb.io.cppm
            src/modules/echter.xgb.reg.cppm
            src/modules/echter.xgb.cppm
)

target_include_directories(Echter_XGBoost
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(Echter_XGBoost
    PRIVATE
        CUDA::cudart
        cudf::cudf
)

set_target_properties(Echter_XGBoost PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_SEPARABLE_COMPILATION ON
)

if(ECHTER_XGB_BUILD_TESTS)
    enable_testing()
    add_executable(echter_xgb_test tests/main.cpp)
    target_link_libraries(echter_xgb_test PRIVATE Echter_XGBoost)
    if(ECHTER_XGB_TEST_MODEL)
        add_test(NAME echter_xgb_runtime_test COMMAND echter_xgb_test "${ECHTER_XGB_TEST_MODEL}")
    endif()
endif()

if(ECHTER_XGB_BUILD_PREDICT)
    add_executable(echter_xgb_predict apps/predict.cpp)
    target_link_libraries(echter_xgb_predict PRIVATE
        Echter_XGBoost
        cudf::cudf
        CUDA::cudart
    )
endif()

if(ECHTER_XGB_BUILD_EXAMPLES)
    add_executable(echter_xgb_csv_prediction examples/csv_prediction.cpp)
    target_link_libraries(echter_xgb_csv_prediction PRIVATE Echter_XGBoost)

    add_executable(echter_xgb_cpu_prediction examples/cpu_prediction.cpp)
    target_link_libraries(echter_xgb_cpu_prediction PRIVATE Echter_XGBoost)

    add_executable(echter_xgb_cuda_prediction examples/cuda_prediction.cpp)
    target_link_libraries(echter_xgb_cuda_prediction PRIVATE
        Echter_XGBoost
        CUDA::cudart
    )

    add_executable(echter_xgb_cudf_prediction examples/cudf_prediction.cpp)
    target_link_libraries(echter_xgb_cudf_prediction PRIVATE
        Echter_XGBoost
        cudf::cudf
        CUDA::cudart
    )
endif()
