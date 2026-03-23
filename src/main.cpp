#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "asum_kernel.cuh"
#include "gemm_kernel.cuh"
#include "transpose_kernel.cuh"

// params for gen test data
#define ROW_NUM 1
#define COL_NUM (1024 * 1024 * 128)
#define MID_NUM 1024
#define VALUE_MAX 100.0f
#define EPS 1e-4f

bool compare_result(float a, float b) {
  return std::isfinite(a) && std::isfinite(b) &&
         fabs(a - b) < EPS * std::max(fabs(a), fabs(b));
}

bool compare_result(float* a, float* b, int N, int M) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (!compare_result(a[i * M + j], b[i * M + j])) {
        return 1;
      }
    }
  }
  return 0;
}

void generate_tset_data(float* a, int N, int M) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      a[i * M + j] = float(rand()) / RAND_MAX * VALUE_MAX - (VALUE_MAX / 2);
    }
  }
}

class exit_guard {
 public:
  ~exit_guard() {
    for (std::function<void()> func : funcs) {
      func();
    }
  }
  void Register(std::function<void()> func) { funcs.emplace_back(func); }

 private:
  std::vector<std::function<void()>> funcs;
} global_exit_guard;

#define CHECK_CUDA_WITH_CLEANUP(STAMENT, CLEANUP_FUN)   \
  {                                                     \
    if ((STAMENT) == cudaSuccess) {                     \
      global_exit_guard.Register(CLEANUP_FUN);          \
    } else {                                            \
      std::cout << "exit in " << __LINE__ << std::endl; \
      exit(1);                                          \
    }                                                   \
  }

#define CHECK_CUDA(STAMENT)                             \
  {                                                     \
    if ((STAMENT) == cudaSuccess) {                     \
    } else {                                            \
      std::cout << "exit in " << __LINE__ << std::endl; \
      exit(1);                                          \
    }                                                   \
  }

float a[ROW_NUM][COL_NUM], output, ground_truth = 0;
float *d_a, *d_output;

int main() {
  srand(time(NULL));
  generate_tset_data(a[0], ROW_NUM, COL_NUM);
  CHECK_CUDA_WITH_CLEANUP(cudaMalloc(&d_a, sizeof(a)),
                          []() -> void { cudaFree(d_a); });
  CHECK_CUDA_WITH_CLEANUP(cudaMalloc(&d_output, sizeof(output)),
                          []() -> void { cudaFree(d_output); });
  float time_ms_memcpy_in, time_ms_kernel, time_ms_memcpy_out;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  global_exit_guard.Register([&start, &stop]() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  });
  cudaEventRecord(start);
  CHECK_CUDA(cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_memcpy_in, start, stop);
  cudaEventRecord(start);
  asum(d_a, d_output, COL_NUM);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_kernel, start, stop);
  cudaEventRecord(start);
  CHECK_CUDA(
      cudaMemcpy(&output, d_output, sizeof(output), cudaMemcpyDeviceToHost));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_memcpy_out, start, stop);

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  global_exit_guard.Register(
      [&cublas_handle]() { cublasDestroy(cublas_handle); });
  cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
  float alpha = 1.0f;
  float beta = 0.0f;
  float time_ms_cublas_kernel;
  cudaEventRecord(start);
  cublasSasum(cublas_handle, COL_NUM, d_a, 1, d_output);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_cublas_kernel, start, stop);
  CHECK_CUDA(cudaMemcpy(&ground_truth, d_output, sizeof(output),
                        cudaMemcpyDeviceToHost));

  if (compare_result(output, ground_truth)) {
    std::cout << "ok" << std::endl;
    for (int i = 0; i < 10; ++i) {
      asum(d_a, d_output, COL_NUM);
    }
    time_ms_kernel = 0.0f;
    for (int i = 0; i < 100; ++i) {
      float time_ms;
      cudaEventRecord(start);
      asum(d_a, d_output, COL_NUM);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time_ms, start, stop);
      time_ms_kernel += time_ms;
    }
    time_ms_kernel /= 100;
    for (int i = 0; i < 10; ++i) {
      cublasSasum(cublas_handle, COL_NUM, d_a, 1, d_output);
    }
    time_ms_cublas_kernel = 0.0f;
    for (int i = 0; i < 100; ++i) {
      float time_ms;
      cudaEventRecord(start);
      cublasSasum(cublas_handle, COL_NUM, d_a, 1, d_output);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time_ms, start, stop);
      time_ms_cublas_kernel += time_ms;
    }
    time_ms_cublas_kernel /= 100;
    std::cout << "kernel time : " << time_ms_kernel << "ms, total time: "
              << time_ms_memcpy_in + time_ms_kernel + time_ms_memcpy_out << "ms"
              << std::endl;
    std::cout << "cublas kernel time: " << time_ms_cublas_kernel
              << "ms, rate: " << time_ms_cublas_kernel / time_ms_kernel
              << std::endl;
  } else {
    std::cout << "failed" << std::endl;
  }
  return 0;
}
