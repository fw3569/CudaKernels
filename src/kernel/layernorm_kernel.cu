#include "layernorm_kernel.cuh"

#define BLOCK_SIZE 1024
#define MAX_TILE_SIZE 32

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
    layernorm_kernel(float* a, float* result, int N) {
  __shared__ float s_result[BLOCK_SIZE >> 5];
  __shared__ float s_mean, s_inv_var;
  float ans = 0;
  float reg_a[MAX_TILE_SIZE];
  int base = blockIdx.x * N;
#pragma unroll
  for (int i = 0, pos = threadIdx.x; i < MAX_TILE_SIZE;
       ++i, pos += BLOCK_SIZE) {
    if (pos < N) {
      reg_a[i] = a[base + pos];
      ans += reg_a[i];
    };
  }
  for (int i = 16; i >= 1; i >>= 1) {
    ans += __shfl_xor_sync(0xffffffff, ans, i);
  }
  if ((threadIdx.x & 0x1f) == 0) {
    s_result[threadIdx.x >> 5] = ans;
  }
  __syncthreads();
  if (threadIdx.x < (BLOCK_SIZE >> 5)) {
    ans = s_result[threadIdx.x];
    for (int i = (BLOCK_SIZE >> 6); i >= 1; i >>= 1) {
      ans += __shfl_xor_sync((1ll << (BLOCK_SIZE >> 5)) - 1, ans, i,
                             (BLOCK_SIZE >> 5));
    }
  }
  if (threadIdx.x == 0) {
    s_mean = ans / N;
  }
  __syncthreads();
  float reg_mean = s_mean;
  ans = 0;
#pragma unroll
  for (int i = 0, pos = threadIdx.x; i < MAX_TILE_SIZE;
       ++i, pos += BLOCK_SIZE) {
    if (pos < N) {
      float diff = reg_a[i] - reg_mean;
      ans += diff * diff;
    };
  }
  for (int i = 16; i >= 1; i >>= 1) {
    ans += __shfl_xor_sync(0xffffffff, ans, i);
  }
  if ((threadIdx.x & 0x1f) == 0) {
    s_result[threadIdx.x >> 5] = ans;
  }
  __syncthreads();
  if (threadIdx.x < (BLOCK_SIZE >> 5)) {
    ans = s_result[threadIdx.x];
    for (int i = (BLOCK_SIZE >> 6); i >= 1; i >>= 1) {
      ans += __shfl_xor_sync((1ll << (BLOCK_SIZE >> 5)) - 1, ans, i,
                             (BLOCK_SIZE >> 5));
    }
  }
  if (threadIdx.x == 0) {
    s_inv_var = 1 / sqrtf(ans + 1e-6);
  }
  __syncthreads();
  float reg_inv_var = s_inv_var;
#pragma unroll
  for (int i = 0, pos = threadIdx.x; i < MAX_TILE_SIZE;
       ++i, pos += BLOCK_SIZE) {
    if (pos < N) {
      float tmp = (reg_a[i] - reg_mean) * reg_inv_var;
      result[base + pos] = tmp;
    }
  }
}

#define SM_COUNT 14

void layernorm(float* a, float* result, int N, int M) {
  for (int i = 0; i < N; i += SM_COUNT) {
    layernorm_kernel<<<min(SM_COUNT, N - i), BLOCK_SIZE>>>(a + i * M,
                                                           result + i * M, M);
  }
}
