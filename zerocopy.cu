// when to use mapped pinned memory
// http://stackoverflow.com/questions/5209214/default-pinned-memory-vs-zero-copy-memory
// 
// one interesting fact, this technique is used to let GPU make use of main memory
// basically, no real performance advantage
// in Jetson TK1, this is quite useful due to its limited memory
// http://arrayfire.com/zero-copy-on-tegra-k1/
//
// running this code you will find this technique is not that bad
// result=524800, expected=524800, time(reduce5)=1.491, bandwidth=10.481GB/s
// testMappedPinned: 19.694
// result=524800, expected=524800, time(reduce5)=0.179, bandwidth=87.506GB/s
// testNormal: 16.307

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
struct timeval cpu_timer;

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 
void HandleError(cudaError_t err, const char *file, int line )                     
{                                                                                  
  if (err != cudaSuccess)                                                          
  {                                                                                
    printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line);       
    exit( EXIT_FAILURE );                                                        
  }                                                                                
}

template <unsigned int blockSize>
__global__ void reduce5(int *g_idata, int *g_odata)
{
  extern __shared__ volatile int sdata2[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  sdata2[tid] = g_idata[i] + g_idata[i + blockDim.x];
  __syncthreads();

  // this is just removing for loop by completely unroll
  if (blockSize >= 512) {
    if (tid < 256) { sdata2[tid] += sdata2[tid + 256]; __syncthreads(); }
  }
  if (blockSize >= 256) {
    if (tid < 128) { sdata2[tid] += sdata2[tid + 128]; __syncthreads(); }
  }
  if (blockSize >= 128) {
    if (tid < 64) { sdata2[tid] += sdata2[tid + 64]; __syncthreads(); }
  }

  if (tid < 32) {
    if (blockSize >= 64) sdata2[tid] += sdata2[tid + 32];
    if (blockSize >= 32) sdata2[tid] += sdata2[tid + 16];
    if (blockSize >= 16) sdata2[tid] += sdata2[tid + 8]; 
    if (blockSize >= 8) sdata2[tid] += sdata2[tid + 4]; 
    if (blockSize >= 4) sdata2[tid] += sdata2[tid + 2]; 
    if (blockSize >= 2) sdata2[tid] += sdata2[tid + 1]; 
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata2[0];
}

int N = 1024 * 1024 * 4;
int bytes = N * sizeof(int); // when type changes, bytes easily cahnges
int threadsPerBlock = 1024;
int blocksPerGrid = N / threadsPerBlock;
const int expectedResult = (1 + threadsPerBlock) * threadsPerBlock / 2;

void testMappedPinned()
{
  int *in, *d_in, *out, *d_out;
  HANDLE_ERROR( cudaHostAlloc(&in, bytes, cudaHostAllocMapped) );
  HANDLE_ERROR( cudaHostAlloc(&out, blocksPerGrid * sizeof(int), cudaHostAllocMapped) );

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time;

  // 1, ..., 256
  for (int i = 0; i < N; ++i) in[i] = i % threadsPerBlock + 1;
  // map memory
  HANDLE_ERROR( cudaHostGetDevicePointer(&d_in, in, 0) );
  HANDLE_ERROR( cudaHostGetDevicePointer(&d_out, out, 0) );

  {
    cudaEventRecord(start, 0);
    // here half threads
    switch (threadsPerBlock / 2) {
      case 512: reduce5<512><<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out); break;
      case 256: reduce5<256><<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out); break;
      case 128: reduce5<128><<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out); break;
      case 64: reduce5<64><<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out); break;
      case 32: reduce5<32><<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out); break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("result=%d, expected=%d, time(reduce5)=%.3f, bandwidth=%.3fGB/s\n", 
            out[0], expectedResult, time, N * sizeof(int) * 1000.0 / (1024 * 1024 * 1024 * time));

    for (int i = 0; i < blocksPerGrid; ++i ) {
      if(out[i] != expectedResult) {
        printf("diff found, i=%d, out[i]=%d\n", i, out[i]);
      }
    }
  }

  HANDLE_ERROR(cudaFreeHost(in));
  HANDLE_ERROR(cudaFreeHost(out));
  HANDLE_ERROR( cudaDeviceSynchronize() );
}

void testNormal()
{
  int *in, *out;
  int *d_in, *d_out;

  in = (int *) calloc(N, sizeof(int));
  out = (int *) calloc(blocksPerGrid, sizeof(int));

  HANDLE_ERROR( cudaMalloc(&d_in, N * sizeof(int)) );
  HANDLE_ERROR( cudaMalloc(&d_out, blocksPerGrid * sizeof(int)) );

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time;

  // 1, ..., 256
  for (int i = 0; i < N; ++i) in[i] = i % threadsPerBlock + 1;

  {
    HANDLE_ERROR(cudaMemcpy(d_in, in, N * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(start, 0);
    // here half threads
    switch (threadsPerBlock / 2) {
      case 512: reduce5<512><<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out); break;
      case 256: reduce5<256><<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out); break;
      case 128: reduce5<128><<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out); break;
      case 64: reduce5<64><<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out); break;
      case 32: reduce5<32><<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out); break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    HANDLE_ERROR(cudaMemcpy(out, d_out, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventElapsedTime(&time, start, stop);
    printf("result=%d, expected=%d, time(reduce5)=%.3f, bandwidth=%.3fGB/s\n", 
            out[0], expectedResult, time, N * sizeof(int) * 1000.0 / (1024 * 1024 * 1024 * time));

    for (int i = 0; i < blocksPerGrid; ++i ) {
      if(out[i] != expectedResult) {
        printf("diff found, i=%d, out[i]=%d\n", i, out[i]);
      }
    }
  }

  HANDLE_ERROR(cudaFree(d_in));
  HANDLE_ERROR(cudaFree(d_out));
  free(in);
  free(out);

  HANDLE_ERROR( cudaDeviceSynchronize() );
}

int main()
{
  // configure mapped pinned memory, enable page-locked memory mapping
  // configure this before calling any cuda function
  // otherwise you can get "cannot set while device is active in this process"
  HANDLE_ERROR( cudaSetDeviceFlags(cudaDeviceMapHost) );

  // warm up
  cudaFree(0);

  gettimeofday(&cpu_timer, NULL);
  testMappedPinned();
  {                                                                           
    double p = cpu_timer.tv_sec * 1000.0 + cpu_timer.tv_usec / 1000.0;        
    gettimeofday(&cpu_timer, NULL);                                           
    double c = cpu_timer.tv_sec * 1000.0 + cpu_timer.tv_usec / 1000.0;        
    printf("testMappedPinned: %.3lf\n", c-p);                                                   
  }

  gettimeofday(&cpu_timer, NULL);
  testNormal();
  {                                                
    double p = cpu_timer.tv_sec * 1000.0 + cpu_timer.tv_usec / 1000.0;        
    gettimeofday(&cpu_timer, NULL);                                           
    double c = cpu_timer.tv_sec * 1000.0 + cpu_timer.tv_usec / 1000.0;        
    printf("testNormal: %.3lf\n", c-p); 
  }

  return 0;
}
