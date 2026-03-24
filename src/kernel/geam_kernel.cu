#include "geam_kernel.cuh"

#define BLOCK_SIZE 256

__global__ void geam_kernel(float* a, float* b, float* c, int N) {
  int id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (id < N) {
    c[id] = a[id] + b[id];
  }
}

void geam(float* a, float* b, float* c, int N, int M) {
  dim3 grid((N * M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);
  geam_kernel<<<grid, block>>>(a, b, c, N * M);
}
