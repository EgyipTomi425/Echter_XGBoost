cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(Echter_XGBoost_benchmarks LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(cudf CONFIG REQUIRED)
add_subdirectory(../.. echter_xgboost)

add_executable(echter_benchmark echter_benchmark.cpp)
target_include_directories(echter_benchmark PRIVATE ../../src/regression)
target_link_libraries(echter_benchmark PRIVATE Echter_XGBoost::Echter_XGBoost cudf::cudf CUDA::cudart)

add_executable(xgboost_capi_benchmark xgboost_capi_benchmark.cpp)
target_compile_features(xgboost_capi_benchmark PRIVATE cxx_std_20)
target_link_libraries(xgboost_capi_benchmark PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
