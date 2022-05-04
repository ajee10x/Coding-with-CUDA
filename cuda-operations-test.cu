// ajee10x
/*
(MIN / MAX searches, dividing, multiplying) 
Operations for large array sizes in a serial form on the CPU and parallel on the GPU.
*/

#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 
#include <iostream> 
#include <stdio.h>
#include <math.h>

void sum(float * a, float * b, float * c, int n);

//draw ، runs in parallel on a large number of threads
__global__ void sumKernel(float * a, float * b, float * c) {
  //Global Thread Index in Grid
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //Sum the data corresponding to the given thread
  c[idx] = a[idx] + b[idx];
}

int main(int argc, char * argv[]) {
  const int n = 10000;
  float * a, * b, * c;
  a = new float[n];
  b = new float[n];
  c = new float[n];
  for (int i = 0; i < n; i++) {
    a[i] = rand() / (float) RAND_MAX - 0.5 f;
    b[i] = rand() / (float) RAND_MAX - 0.5 f;
  }
  sum(a, b, c, n);
  for (int i = 0; i < n; i++) {
    printf("c[%d]=%f\n", i, c[i]);
  }
  return 0;
}

void sum(float * a, float * b, float * c, int n) {
  //Total array size in bytes
  int numBytes = n * sizeof(float);

  //GPU pointer declaration
  float * aDev = NULL;
  float * bDev = NULL;
  float * cDev = NULL;

  float gpuTime = 0.0 f;
  cudaEvent_t start, stop;

  //GPU memory allocation
  cudaMalloc((void ** ) & aDev, numBytes);
  cudaMalloc((void ** ) & bDev, numBytes);
  cudaMalloc((void ** ) & cDev, numBytes);
  //Determining the number of blocks in a grid and threads in a block
  dim3 threads = dim3(512);
  dim3 blocks = dim3(n / threads.x);
  //Copying input data from CPU memory to GPU memory
  cudaMemcpy(aDev, a, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(bDev, b, numBytes, cudaMemcpyHostToDevice);

  cudaEventCreate( & start);
  cudaEventCreate( & stop);
  cudaEventRecord(start, 0);

  //Running the summation kernel on the GPU with a given configuration of blocks and threads
  sumKernel << < blocks, threads >> > (aDev, bDev, cDev);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime( & gpuTime, start, stop);
  cudaEventRecord(start, 0);
  printf("time in miliseconds is %.4f\n", gpuTime);

  //Copy result from GPU to CPU
  cudaMemcpy(c, cDev, numBytes, cudaMemcpyDeviceToHost);
  //Freeing memory on the GPU
  cudaFree(aDev);
  cudaFree(bDev);
  cudaFree(cDev);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
