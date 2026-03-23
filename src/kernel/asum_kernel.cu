#include "asum_kernel.cuh"

#define BLOCK_SIZE 256
#define TILE_SIZE 32

__global__ void asum_kernel(float* a, float* result, int N) {
  __shared__ float s_result[BLOCK_SIZE >> 5];
  float ans = 0;
  for (int i = 0, pos = blockIdx.x * BLOCK_SIZE * TILE_SIZE + threadIdx.x;
       i < TILE_SIZE && pos < N; ++i, pos += BLOCK_SIZE) {
    ans += abs(a[pos]);
  }
  for (int i = 16; i >= 1; i >>= 1) {
    ans += __shfl_xor_sync(0xffffffff, ans, i, 32);
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
  } else {
    return;
  }
  if (threadIdx.x == 0) {
    atomicAdd(result, ans);
  }
}

void asum(float* a, float* result, int N) {
  cudaMemset(result, 0, sizeof(float));
  dim3 grid((N + BLOCK_SIZE * TILE_SIZE - 1) / (BLOCK_SIZE * TILE_SIZE));
  dim3 block(BLOCK_SIZE);
  asum_kernel<<<grid, block>>>(a, result, N);
}
