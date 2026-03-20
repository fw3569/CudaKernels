#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "gemm_kernel.cuh"
#include "transpose_kernel.cuh"

// params for gen test data
#define ROW_NUM 4096
#define COL_NUM 4096
#define MID_NUM 1024
#define VALUE_MAX 100.0f

int compare_result(float* c, float* ground_truth, int N, int M) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (!std::isfinite(c[i * M + j]) ||
          !std::isfinite(ground_truth[i * M + j]) ||
          fabs(c[i * M + j] - ground_truth[i * M + j]) >=
              1e-4 *
                  std::max(fabs(c[i * M + j]), fabs(ground_truth[i * M + j]))) {
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

float a[ROW_NUM][COL_NUM], b[ROW_NUM][COL_NUM], c[COL_NUM][ROW_NUM],
    ground_truth[COL_NUM][ROW_NUM];
float *dev_a, *dev_b, *dev_c;

int main() {
  srand(time(NULL));
  generate_tset_data(a[0], ROW_NUM, COL_NUM);
  CHECK_CUDA_WITH_CLEANUP(cudaMalloc(&dev_a, sizeof(a)),
                          []() -> void { cudaFree(dev_a); });
  CHECK_CUDA_WITH_CLEANUP(cudaMalloc(&dev_b, sizeof(b)),
                          []() -> void { cudaFree(dev_b); });
  CHECK_CUDA_WITH_CLEANUP(cudaMalloc(&dev_c, sizeof(c)),
                          []() -> void { cudaFree(dev_c); });
  float time_ms_memcpy_in, time_ms_kernel, time_ms_memcpy_out;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  global_exit_guard.Register([&start, &stop]() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  });
  cudaEventRecord(start);
  CHECK_CUDA(cudaMemcpy(dev_a, a, sizeof(a), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dev_b, 0, sizeof(b)));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_memcpy_in, start, stop);
  cudaEventRecord(start);
  transpose(dev_a, dev_c, ROW_NUM, COL_NUM);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_kernel, start, stop);
  cudaEventRecord(start);
  CHECK_CUDA(cudaMemcpy(c, dev_c, sizeof(c), cudaMemcpyDeviceToHost));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_memcpy_out, start, stop);

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  float alpha = 1.0f;
  float beta = 0.0f;
  float time_ms_cublas_kernel;
  cudaEventRecord(start);
  cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, ROW_NUM, COL_NUM, &alpha,
              dev_a, COL_NUM, &beta, dev_b, ROW_NUM, dev_c, ROW_NUM);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_cublas_kernel, start, stop);
  CHECK_CUDA(cudaMemcpy(ground_truth, dev_c, sizeof(ground_truth),
                        cudaMemcpyDeviceToHost));

  if (compare_result(c[0], ground_truth[0], COL_NUM, ROW_NUM) == 0) {
    std::cout << "ok" << std::endl;
    for (int i = 0; i < 10; ++i) {
      transpose(dev_a, dev_c, ROW_NUM, COL_NUM);
    }
    time_ms_kernel = 0.0f;
    for (int i = 0; i < 100; ++i) {
      float time_ms;
      cudaEventRecord(start);
      transpose(dev_a, dev_c, ROW_NUM, COL_NUM);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time_ms, start, stop);
      time_ms_kernel += time_ms;
    }
    time_ms_kernel /= 100;
    for (int i = 0; i < 10; ++i) {
      cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, ROW_NUM, COL_NUM,
                  &alpha, dev_a, COL_NUM, &beta, dev_b, ROW_NUM, dev_c,
                  ROW_NUM);
    }
    time_ms_cublas_kernel = 0.0f;
    for (int i = 0; i < 100; ++i) {
      float time_ms;
      cudaEventRecord(start);
      cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, ROW_NUM, COL_NUM,
                  &alpha, dev_a, COL_NUM, &beta, dev_b, ROW_NUM, dev_c,
                  ROW_NUM);
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
