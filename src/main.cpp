#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <device_launch_parameters.h>

#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "asum_kernel.cuh"
#include "geam_kernel.cuh"
#include "gemm_kernel.cuh"
#include "softmax_kernel.cuh"
#include "transpose_kernel.cuh"

// params for gen test data
#define ROW_NUM 1
#define COL_NUM (1024 * 32)
#define MID_NUM 1024
#define VALUE_MAX 100.0f
#define EPS 1e-4f

bool compare_result(float a, float b) {
  return std::isfinite(a) && std::isfinite(b) &&
         fabs(a - b) <= EPS * std::max(fabs(a), fabs(b));
}

bool compare_result(float* a, float* b, int N, int M) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (!compare_result(a[i * M + j], b[i * M + j])) {
        return false;
      }
    }
  }
  return true;
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

float a[ROW_NUM][COL_NUM], b[ROW_NUM][COL_NUM], output[ROW_NUM][COL_NUM],
    ground_truth[ROW_NUM][COL_NUM];
float *d_a, *d_b, *d_output;

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
  global_exit_guard.Register([]() { softmaxDestroy(); });
  cudaEventRecord(start);
  softmax(d_a, d_output, COL_NUM);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_kernel, start, stop);
  cudaEventRecord(start);
  CHECK_CUDA(
      cudaMemcpy(output, d_output, sizeof(output), cudaMemcpyDeviceToHost));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_memcpy_out, start, stop);

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);
  global_exit_guard.Register([&cudnn]() { cudnnDestroy(cudnn); });
  cudnnTensorDescriptor_t cudnn_desc;
  cudnnCreateTensorDescriptor(&cudnn_desc);
  global_exit_guard.Register(
      [&cudnn_desc]() { cudnnDestroyTensorDescriptor(cudnn_desc); });
  cudnnSetTensor4dDescriptor(cudnn_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                             COL_NUM, 1, 1);
  float alpha = 1.0f;
  float beta = 0.0f;
  float time_ms_cublas_kernel;
  cudaEventRecord(start);
  cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
                      &alpha, cudnn_desc, d_a, &beta, cudnn_desc, d_output);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms_cublas_kernel, start, stop);
  CHECK_CUDA(cudaMemcpy(ground_truth, d_output, sizeof(ground_truth),
                        cudaMemcpyDeviceToHost));

  if (compare_result((float*)output, (float*)ground_truth, ROW_NUM, COL_NUM)) {
    std::cout << "ok" << std::endl;
    for (int i = 0; i < 10; ++i) {
      softmax(d_a, d_output, COL_NUM);
    }
    time_ms_kernel = 0.0f;
    for (int i = 0; i < 100; ++i) {
      float time_ms;
      cudaEventRecord(start);
      softmax(d_a, d_output, COL_NUM);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time_ms, start, stop);
      time_ms_kernel += time_ms;
    }
    time_ms_kernel /= 100;
    for (int i = 0; i < 10; ++i) {
      cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_FAST,
                          CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, cudnn_desc, d_a,
                          &beta, cudnn_desc, d_output);
    }
    time_ms_cublas_kernel = 0.0f;
    for (int i = 0; i < 100; ++i) {
      float time_ms;
      cudaEventRecord(start);
      cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_FAST,
                          CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, cudnn_desc, d_a,
                          &beta, cudnn_desc, d_output);
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
