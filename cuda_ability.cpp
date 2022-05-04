#include <stdio.h>
#include <cuda_runtime_api.h>

int main()
{
  int deviceCount;
  cudaDeviceProp deviceProp;

  // the number of CUDA-capable devices attached to this system
  cudaGetDeviceCount(&deviceCount);
  printf("Device count: %d\n\n", deviceCount);
  for (int i = 0; i < deviceCount; i++)
  {
   // receiving information about the compute-device
    cudaGetDeviceProperties(&deviceProp, i);
 // Displays device information
    printf("Device name: %s\n", deviceProp.name);
    printf("Total global memory: %d\n", deviceProp.totalGlobalMem);
    
    printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
      deviceProp.maxThreadsDim[0],
      deviceProp.maxThreadsDim[1],
      deviceProp.maxThreadsDim[2]);

    printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
  }
  return 0;
}
