// memory page is managed by OS
// "unpageable" means cuda directly accesses physical RAM
// to access pageable memory allocated by like malloc()
//   pageable memory -> pinned memory -> device memory
// if we directly operate on pinned memory
//   then we save transfer from pageable memory to pinned memory

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
void HandleError(cudaError_t err, const char *file, int line )
{
  if (err != cudaSuccess)
  {
    printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit( EXIT_FAILURE );
  }
}

int main()
{
  unsigned int N = 256 * 1024 * 1024;

  float *d = NULL;
  HANDLE_ERROR( cudaMalloc((void**)&d, N * sizeof(float)) );
  
  cudaEvent_t startEvent, stopEvent;
  HANDLE_ERROR( cudaEventCreate(&startEvent) );
  HANDLE_ERROR( cudaEventCreate(&stopEvent) );

  float *h_a = NULL, *h_b = NULL;
  float time = 0;

  // test on pageable memory
  h_a = (float *) malloc(N * sizeof(float));
  h_b = (float *) malloc(N * sizeof(float));

  HANDLE_ERROR( cudaEventRecord(startEvent, 0) );
  HANDLE_ERROR( cudaMemcpy(d, h_a, N * sizeof(float), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaEventRecord(stopEvent, 0) );
  HANDLE_ERROR( cudaEventSynchronize(stopEvent) );
  HANDLE_ERROR( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("Pageable, host to device, %.3fGB/s\n", 
          N * sizeof(float) * 1000.0 / (1024 * 1024 * 1024 * time));

  HANDLE_ERROR( cudaEventRecord(startEvent, 0) );
  HANDLE_ERROR( cudaMemcpy(h_b, d, N * sizeof(float), cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaEventRecord(stopEvent, 0) );
  HANDLE_ERROR( cudaEventSynchronize(stopEvent) );
  HANDLE_ERROR( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("Pageable, device to host, %.3fGB/s\n", 
          N * sizeof(float) * 1000.0 / (1024 * 1024 * 1024 * time));

  free(h_a);
  free(h_b);
  h_a = NULL; 
  h_b = NULL;

  // test on pinned memory
  HANDLE_ERROR( cudaMallocHost((void**)&h_a, N * sizeof(float)) );
  HANDLE_ERROR( cudaMallocHost((void**)&h_b, N * sizeof(float)) );

  HANDLE_ERROR( cudaEventRecord(startEvent, 0) );
  HANDLE_ERROR( cudaMemcpy(d, h_a, N * sizeof(float), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaEventRecord(stopEvent, 0) );
  HANDLE_ERROR( cudaEventSynchronize(stopEvent) );
  HANDLE_ERROR( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("Pinned, host to device, %.3fGB/s\n", 
          N * sizeof(float) * 1000.0 / (1024 * 1024 * 1024 * time));

  HANDLE_ERROR( cudaEventRecord(startEvent, 0) );
  HANDLE_ERROR( cudaMemcpy(h_b, d, N * sizeof(float), cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaEventRecord(stopEvent, 0) );
  HANDLE_ERROR( cudaEventSynchronize(stopEvent) );
  HANDLE_ERROR( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("Pinned, device to host, %.3fGB/s\n", 
          N * sizeof(float) * 1000.0 / (1024 * 1024 * 1024 * time));

  cudaFreeHost(h_a);
  cudaFreeHost(h_b);


  // clean up
  cudaFree(d);
  HANDLE_ERROR( cudaEventDestroy(startEvent) );
  HANDLE_ERROR( cudaEventDestroy(stopEvent) );  
  return 0;
}
