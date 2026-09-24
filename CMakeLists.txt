cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(Echter_XGBoost VERSION 0.3.0 LANGUAGES CXX CUDA)

option(ECHTER_XGB_BUILD_TESTS "Build test executable" ${PROJECT_IS_TOP_LEVEL})
option(ECHTER_XGB_BUILD_PREDICT "Build CSV prediction executable" ${PROJECT_IS_TOP_LEVEL})
option(ECHTER_XGB_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOP_LEVEL})
set(ECHTER_XGB_TEST_MODEL "" CACHE FILEPATH "Model JSON used by CTest runtime test")

if(PROJECT_IS_TOP_LEVEL)
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}" CACHE PATH "Install prefix" FORCE)
    endif()
endif()

include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

find_package(CUDAToolkit REQUIRED)
find_package(cudf CONFIG REQUIRED)

add_compile_options(
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wall;-Wextra>"
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>"
)

add_library(Echter_XGBoost STATIC
    src/core/cuda_check.cuh
    src/core/cudf_convert.cuh
    src/core/device_buffer.cpp
    src/core/device_buffer.hpp
    src/io/echter.xgb.io.cpp
    src/io/io_backend.cpp
    src/io/io_backend.hpp
    src/modules/echter.xgb.data.cpp
    src/modules/echter.xgb.reg.cpp
    src/regression/regression_cudf_adapter.cpp
    src/regression/regression_cudf_adapter.hpp
    src/regression/regression_kernels.cu
    src/regression/regression_kernels.cuh
    src/regression/regression_kernels.hpp
    src/regression/regression_model.hpp
    src/regression/regression_model_loader.cpp
    src/regression/regression_runtime.cpp
)
add_library(Echter_XGBoost::Echter_XGBoost ALIAS Echter_XGBoost)

target_sources(Echter_XGBoost
    PUBLIC
        FILE_SET CXX_MODULES TYPE CXX_MODULES FILES
            src/modules/echter.cppm
            src/modules/echter.xgb.data.cppm
            src/io/echter.xgb.io.cppm
            src/modules/echter.xgb.reg.cppm
            src/modules/echter.xgb.cppm
)

target_compile_features(Echter_XGBoost PUBLIC cxx_std_20)
set_target_properties(Echter_XGBoost PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(Echter_XGBoost
    PRIVATE
        CUDA::cudart
        cudf::cudf
)

if(ECHTER_XGB_BUILD_TESTS)
    enable_testing()
    add_executable(echter_xgb_test tests/main.cpp)
    target_link_libraries(echter_xgb_test PRIVATE Echter_XGBoost CUDA::cudart)
    add_test(NAME echter_xgb_unit_test COMMAND echter_xgb_test)
    if(ECHTER_XGB_TEST_MODEL)
        add_test(NAME echter_xgb_runtime_test COMMAND echter_xgb_test "${ECHTER_XGB_TEST_MODEL}")
    endif()
    install(TARGETS echter_xgb_test RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/test)
endif()

if(ECHTER_XGB_BUILD_PREDICT)
    add_executable(echter_xgb_predict apps/predict.cpp)
    target_link_libraries(echter_xgb_predict PRIVATE Echter_XGBoost)
    install(TARGETS echter_xgb_predict RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(ECHTER_XGB_BUILD_EXAMPLES)
    foreach(example IN ITEMS cpu csv cuda cudf)
        add_executable(echter_xgb_${example}_prediction examples/${example}_prediction.cpp)
        target_link_libraries(echter_xgb_${example}_prediction PRIVATE Echter_XGBoost CUDA::cudart)
        install(TARGETS echter_xgb_${example}_prediction RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)
    endforeach()
    target_link_libraries(echter_xgb_cudf_prediction PRIVATE cudf::cudf)
endif()
