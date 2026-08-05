cmake_minimum_required(VERSION 3.30.4)

project(CompileIQNVBenchReduction
  DESCRIPTION "CMake-based CompileIQ optimization of an NVBench-instrumented CUDA reduction benchmark"
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT (DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED ENV{CUDAARCHS}))
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Point CMAKE_PREFIX_PATH at an installed NVBench prefix. For cuda-bench wheels:
#
#   cmake -S . -B build -G Ninja \
#     -DCMAKE_PREFIX_PATH="$(python -m cuda.bench.nvbench --prefix)"
#
find_package(nvbench CONFIG REQUIRED)

add_executable(reduction_bench reduction_bench.cu)
target_link_libraries(reduction_bench PRIVATE nvbench::main)

set(COMPILEIQ_PTXAS_CONTROLS "" CACHE FILEPATH "PTXAS controls file")
set(COMPILEIQ_EXTRA_NVCC_OPTIONS "" CACHE STRING "Extra CUDA compiler options")

if(COMPILEIQ_PTXAS_CONTROLS)
  target_compile_options(reduction_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas>
    $<$<COMPILE_LANGUAGE:CUDA>:--apply-controls=${COMPILEIQ_PTXAS_CONTROLS}>
  )
  set_property(SOURCE reduction_bench.cu APPEND PROPERTY
    OBJECT_DEPENDS "${COMPILEIQ_PTXAS_CONTROLS}"
  )
endif()

if(COMPILEIQ_EXTRA_NVCC_OPTIONS)
  separate_arguments(COMPILEIQ_EXTRA_NVCC_OPTIONS_LIST NATIVE_COMMAND
    "${COMPILEIQ_EXTRA_NVCC_OPTIONS}"
  )
  foreach(option IN LISTS COMPILEIQ_EXTRA_NVCC_OPTIONS_LIST)
    target_compile_options(reduction_bench PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${option}>
    )
  endforeach()
endif()

set_target_properties(reduction_bench PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
