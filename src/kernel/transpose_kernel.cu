#include "transpose_kernel.cuh"

#define TILE_SIZE 32

__global__ void transpose_kernel(float* a, float* b, int N, int M) {
  __shared__ float sa[TILE_SIZE][TILE_SIZE + 1];
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  if (row < N && col < M) {
    sa[threadIdx.x][threadIdx.y] = a[row * M + col];
  }
  __syncthreads();
  int colt = blockIdx.y * TILE_SIZE + threadIdx.x;
  int rowt = blockIdx.x * TILE_SIZE + threadIdx.y;
  if (rowt < M && colt < N) {
    b[rowt * N + colt] = sa[threadIdx.y][threadIdx.x];
  }
}

void transpose(float* a, float* b, int N, int M) {
  dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
  dim3 block(TILE_SIZE, TILE_SIZE);
  transpose_kernel<<<grid, block>>>(a, b, N, M);
}
