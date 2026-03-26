cmake_minimum_required(VERSION 4.0)
set(CMAKE_CUDA_HOST_COMPILER "D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
set(CMAKE_CXX_COMPILER "D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
set(CMAKE_CUDA_COMPILER "nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_COMPILER_FORCED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CUDNN_ROOT "D:/Program Files/NVIDIA/CUDNN/v9.20")
project(Proj LANGUAGES CXX CUDA)
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
add_executable(
  proj
  src/main.cpp
  src/kernel/gemm_kernel.cu
  src/kernel/transpose_kernel.cu
  src/kernel/asum_kernel.cu
  src/kernel/geam_kernel.cu
  src/kernel/softmax_kernel.cu
  src/kernel/layernorm_kernel.cu
)
target_include_directories(
  proj
  PRIVATE
  ./include
  "${CUDNN_ROOT}/include/12.9"
)
find_package(CUDAToolkit REQUIRED)
target_link_directories(
  proj
  PRIVATE
  ${CUDAToolkit_LIBRARY_DIR}
  "${CUDNN_ROOT}/lib/12.9/x64"
)
target_link_libraries(
  proj
  PRIVATE
  CUDA::cudart
  CUDA::cublas
  cudadevrt
  cudnn
)
target_compile_options(
  proj
  PRIVATE
  $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler /Zi>
)
target_link_options(
  proj
  PRIVATE
  $<$<CONFIG:RelWithDebInfo>:-Xlinker /DEBUG>
)
