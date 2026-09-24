cmake_minimum_required(VERSION 3.28)

project(Echter_XGBoost VERSION 0.3.0 LANGUAGES CXX CUDA)

option(ECHTER_XGB_BUILD_TESTS "Build test executable" ON)
option(ECHTER_XGB_BUILD_PREDICT "Build CSV prediction executable" ON)
option(ECHTER_XGB_BUILD_EXAMPLES "Build examples" ON)
set(ECHTER_XGB_TEST_MODEL "" CACHE FILEPATH "Model JSON used by CTest runtime test")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# `cmake --install <build-dir>` places the executables in <source>/bin by
# default. They keep the RPATH to the CUDA and cuDF libraries they link.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}" CACHE PATH "Install prefix" FORCE)
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(cudf CONFIG REQUIRED)

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")

add_library(Echter_XGBoost STATIC
    src/core/cuda_utils.cuh
    src/core/cudf_convert.cuh
    src/core/model_backend.cu
    src/core/model_backend.hpp
    src/io/echter.xgb.io.cpp
    src/io/io_backend.cu
    src/io/io_backend.hpp
    src/modules/echter.xgb.cpp
    src/modules/echter.xgb.data.cpp
    src/modules/echter.xgb.reg.cpp
    src/regression/regression_backend.hpp
    src/regression/regression_cudf_adapter.cpp
    src/regression/regression_cudf_adapter.hpp
    src/regression/regression_kernels.cu
    src/regression/regression_kernels.cuh
    src/regression/regression_model.cuh
    src/regression/regression_model_loader.cpp
    src/regression/regression_runtime.cu
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
)

set(ECHTER_XGB_EXECUTABLES)

if(ECHTER_XGB_BUILD_TESTS)
    enable_testing()
    add_executable(echter_xgb_test tests/main.cpp)
    target_link_libraries(echter_xgb_test PRIVATE
        Echter_XGBoost
        CUDA::cudart
    )
    add_test(NAME echter_xgb_unit_test COMMAND echter_xgb_test)
    if(ECHTER_XGB_TEST_MODEL)
        add_test(NAME echter_xgb_runtime_test COMMAND echter_xgb_test "${ECHTER_XGB_TEST_MODEL}")
    endif()
    list(APPEND ECHTER_XGB_EXECUTABLES echter_xgb_test)
endif()

if(ECHTER_XGB_BUILD_PREDICT)
    add_executable(echter_xgb_predict apps/predict.cpp)
    target_link_libraries(echter_xgb_predict PRIVATE Echter_XGBoost)
    list(APPEND ECHTER_XGB_EXECUTABLES echter_xgb_predict)
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

    list(APPEND ECHTER_XGB_EXECUTABLES
        echter_xgb_csv_prediction
        echter_xgb_cpu_prediction
        echter_xgb_cuda_prediction
        echter_xgb_cudf_prediction
    )
endif()

install(TARGETS ${ECHTER_XGB_EXECUTABLES} RUNTIME DESTINATION bin)
