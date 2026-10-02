cmake_minimum_required(VERSION 3.24)

# bnk: a V100 (sm_70) + host-RAM MoE inference engine for qwen4exp models.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(bnk LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ---- vendored ggml: CPU backend only (quant formats, AVX2 dot products, gguf parser)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

set(BNK_CUDA_FLAGS -lineinfo --use_fast_math -Xptxas=-v -Wno-deprecated-gpu-targets
    --expt-relaxed-constexpr -diag-suppress=177,550)
set(BNK_CUDA_FLAGS_QUIET -lineinfo --use_fast_math -Wno-deprecated-gpu-targets
    --expt-relaxed-constexpr -diag-suppress=177,550)

add_library(bnk_core STATIC
  src/core/gguf_model.cpp
  src/core/model.cpp
  src/core/expert_cache.cpp
  src/cpu/expert_pool.cpp
  src/cpu/kernels.cpp
  src/kernels/quant_gemv.cu
  src/kernels/gemv_r.cu
  src/kernels/rfmt.cpp
  src/kernels/ops.cu
  src/kernels/moe.cu
  src/kernels/qsa.cu
  src/engine/engine.cpp
  src/engine/mtp.cpp
  src/engine/generate.cpp
  src/engine/sampler.cpp
  src/server/serve.cpp
  src/engine/prefill.cpp
  src/kernels/gemm.cu
)
target_include_directories(bnk_core PUBLIC src third_party/ggml/include third_party/ggml/src)
target_link_libraries(bnk_core PUBLIC ggml ggml-base ggml-cpu CUDA::cudart CUDA::cublas CUDA::cuda_driver Threads::Threads)
target_compile_options(bnk_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${BNK_CUDA_FLAGS_QUIET}>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wno-unused-function>)
set_target_properties(bnk_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(bnk tools/bnk_main.cpp)
target_link_libraries(bnk PRIVATE bnk_core)
target_compile_options(bnk PRIVATE -O2 -march=native)


option(BNK_BUILD_TESTS "build kernel tests" ON)
if(BNK_BUILD_TESTS)
  add_executable(test_dequant tests/test_dequant.cu)
  target_link_libraries(test_dequant PRIVATE bnk_core)
  target_compile_options(test_dequant PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${BNK_CUDA_FLAGS_QUIET}>)
endif()

add_executable(bench_cpu_experts tests/bench_cpu_experts.cpp)
target_link_libraries(bench_cpu_experts PRIVATE bnk_core)
add_executable(test_cpu_kernels tests/test_cpu_kernels.cpp)
target_link_libraries(test_cpu_kernels PRIVATE bnk_core)
add_executable(bench_gemv tests/bench_gemv.cu)
target_link_libraries(bench_gemv PRIVATE bnk_core)
target_compile_options(bench_gemv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${BNK_CUDA_FLAGS_QUIET}>)
