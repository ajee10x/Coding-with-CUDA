// ajee10x
//The solution of the matrix multiplication problem on a GPU and CUDA technology
#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdlib.h>

#include <vector>

#include "time.h"

#define N(10)
#define IDX2C(i, j, ld)(((j) * (ld)) + (i))

void printMatrix(char * name, float * matrix, int width, int height) {
  printf("%s:\n", name);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      printf("%10.6f", matrix[IDX2C(i, j, width)]);
    }
    printf("\n");
  }
}
__global__ void mult(float * a, float * b, float * c) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  float sum = 0.0;

  int ia = N * 128 * by + N * ty;
  int ib = 128 * bx + tx;

  for (int k = 0; k < N; k++)
    sum += a[ia + k] * b[ib + k * N];
  int ic = N * 128 * by + 128 * bx;
  c[ic + N * ty + tx] = sum;
}
int main(void) {
  float * a;
  float * b;
  float * c;

  a = (float * ) malloc(N * N * sizeof(float));
  b = (float * ) malloc(N * N * sizeof(float));
  c = (float * ) malloc(N * N * sizeof(float));

  float * dev_a, * dev_b;
  float * dev_c;

  float gpuTime = 0.0 f;
  cudaEvent_t start, stop;

  cudaMalloc((float ** ) & dev_a, N * N * sizeof(float));
  cudaMalloc((float ** ) & dev_b, N * N * sizeof(float));
  cudaMalloc((float ** ) & dev_c, N * N * sizeof(float));

  // 
  int i, j;
  for (j = 0; j < N; j++) {
    for (i = 0; i < N; i++) {
      a[IDX2C(i, j, N)] = rand() / (float) RAND_MAX;
    }
  }
  printMatrix("a", a, N, N);
  for (j = 0; j < N; j++) {
    for (i = 0; i < N; i++) {
      b[IDX2C(i, j, N)] = rand() / (float) RAND_MAX;
    }
  }
  printMatrix("b", b, N, N);
  for (j = 0; j < N; j++) {
    for (i = 0; i < N; i++) {
      c[IDX2C(i, j, N)] = 0.0 f;
    }
  }
  printMatrix("c", c, N, N);

  cudaMemcpy(dev_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventCreate( & start);
  cudaEventCreate( & stop);
  cudaEventRecord(start, 0);

  mult << < max(1, N / 128), 128 >> > (dev_a, dev_b, dev_c);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime( & gpuTime, start, stop);
  cudaEventRecord(start, 0);

  cudaMemcpy(c, dev_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  printMatrix("c after mult", c, N, N);
  printf("time in miliseconds is %.4f\n", gpuTime);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(a);
  free(b);
  free(c);
  return 0;
}
