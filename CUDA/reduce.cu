#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 
void HandleError(cudaError_t err, const char *file, int line )                     
{                                                                                  
  if (err != cudaSuccess)                                                          
  {                                                                                
    printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line);       
    exit( EXIT_FAILURE );                                                        
  }                                                                                
}

__global__ void reduce0(int *g_idata, int *g_odata)
{
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  // 0  1  2  3  4  5  6  7 --- thread id
  // |__|  |__|  |__|  |__| --- 0, 2, 4, 6 (+1)
  // |     |     |     |
  // |_____|     |_____|    --- 0, 4 (+2)
  // |           |
  // |___________|          --- 0 (+4)
  // |
  // #

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (0 == tid % (2 * s)) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce1(int *g_idata, int *g_odata)
{
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  // 0  1  2  3  4  5  6  7 --- thread id
  // |__|  |__|  |__|  |__| --- 0, 1, 2, 3 (+1)
  // |     |     |     |
  // |_____|     |_____|    --- 0, 1 (+2)
  // |           |
  // |___________|          --- 0 (+4)
  // |
  // #
  // because threads are executed in warps
  // in this way, some warps can run the same path
  // in reduce0, all warps can't 

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce2(int *g_idata, int *g_odata)
{
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  // 0  1  2  3  4  5  6  7 --- thread id
  // |___________|  |  |  |
  //    |___________|  |  |
  //       |___________|  |
  //          |___________| --- 0, 1, 2, 3 (+4)
  // 
  // |_____|    
  //    |_____|             --- 0, 1 (+2)
  //
  // |__|                   --- 0 (+1)
  // |
  // #

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce3(int *g_idata, int *g_odata)
{
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  // half of threads in each block will be wasted in the 1st iteration of next for loop
  // reduce number of threads and make them do more work before loop
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce4(int *g_idata, int *g_odata)
{
  // add volatile to ensure implicit data synchronisation within the warp
  // http://stackoverflow.com/questions/13994178/cuda-how-to-unroll-first-32-threads-so-they-will-be-executed-in-parallel
  extern __shared__ volatile int sdata2[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  sdata2[tid] = g_idata[i] + g_idata[i + blockDim.x];
  __syncthreads();

  // make the the first warp finish the rest work
  // other warps can have a rest
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata2[tid] += sdata2[tid + s];
    }
    __syncthreads();
  }

  // threads in the same warp always execute the same instructions, no need to sync
  // you can't put __syncthreads() in if, how can a whole block sync there?
  if (tid < 32) {
    sdata2[tid] += sdata2[tid + 32];
    sdata2[tid] += sdata2[tid + 16];
    sdata2[tid] += sdata2[tid + 8]; 
    sdata2[tid] += sdata2[tid + 4]; 
    sdata2[tid] += sdata2[tid + 2]; 
    sdata2[tid] += sdata2[tid + 1]; 
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata2[0];
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

int main()
{
  printf("performance improvement\n");
  int *in, *out;
  int *d_in, *d_out;

  int N = 1024 * 1024 * 4 ;

  int threadsPerBlock = 1024;
  int blocksPerGrid = N / threadsPerBlock;
  const int expectedResult = (1 + threadsPerBlock) * threadsPerBlock / 2;

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

  // test on reduce
  {
    HANDLE_ERROR(cudaMemcpy(d_in, in, N * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(start, 0);
    reduce0<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_in, d_out);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    HANDLE_ERROR(cudaMemcpy(out, d_out, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventElapsedTime(&time, start, stop);
    printf("result=%d, expected=%d, time(reduce0)=%.3f, bandwidth=%.3fGB/s\n", 
            out[0], expectedResult, time, N * sizeof(int) * 1000.0 / (1024 * 1024 * 1024 * time));

    for (int i = 0; i < blocksPerGrid; ++i ) {
      if(out[i] != expectedResult) {
        printf("diff found, i=%d, out[i]=%d\n", i, out[i]);
      }
    }
  }

  {
    HANDLE_ERROR(cudaMemcpy(d_in, in, N * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(start, 0);
    reduce1<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_in, d_out);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    HANDLE_ERROR(cudaMemcpy(out, d_out, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventElapsedTime(&time, start, stop);
    printf("result=%d, expected=%d, time(reduce1)=%.3f, bandwidth=%.3fGB/s\n", 
            out[0], expectedResult, time, N * sizeof(int) * 1000.0 / (1024 * 1024 * 1024 * time));

    for (int i = 0; i < blocksPerGrid; ++i ) {
      if(out[i] != expectedResult) {
        printf("diff found, i=%d, out[i]=%d\n", i, out[i]);
      }
    }
  }

  {
    HANDLE_ERROR(cudaMemcpy(d_in, in, N * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(start, 0);
    reduce2<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_in, d_out);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    HANDLE_ERROR(cudaMemcpy(out, d_out, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventElapsedTime(&time, start, stop);
    printf("result=%d, expected=%d, time(reduce2)=%.3f, bandwidth=%.3fGB/s\n", 
            out[0], expectedResult, time, N * sizeof(int) * 1000.0 / (1024 * 1024 * 1024 * time));

    for (int i = 0; i < blocksPerGrid; ++i ) {
      if(out[i] != expectedResult) {
        printf("diff found, i=%d, out[i]=%d\n", i, out[i]);
      }
    }
  }

  {
    HANDLE_ERROR(cudaMemcpy(d_in, in, N * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(start, 0);
    // here half threads
    reduce3<<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    HANDLE_ERROR(cudaMemcpy(out, d_out, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventElapsedTime(&time, start, stop);
    printf("result=%d, expected=%d, time(reduce3)=%.3f, bandwidth=%.3fGB/s\n", 
            out[0], expectedResult, time, N * sizeof(int) * 1000.0 / (1024 * 1024 * 1024 * time));

    for (int i = 0; i < blocksPerGrid; ++i ) {
      if(out[i] != expectedResult) {
        printf("diff found, i=%d, out[i]=%d\n", i, out[i]);
      }
    }
  }

  {
    HANDLE_ERROR(cudaMemcpy(d_in, in, N * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(start, 0);
    // here half threads
    reduce4<<<blocksPerGrid, threadsPerBlock / 2, threadsPerBlock / 2 * sizeof(int)>>>(d_in, d_out);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    HANDLE_ERROR(cudaMemcpy(out, d_out, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventElapsedTime(&time, start, stop);
    printf("result=%d, expected=%d, time(reduce4)=%.3f, bandwidth=%.3fGB/s\n", 
            out[0], expectedResult, time, N * sizeof(int) * 1000.0 / (1024 * 1024 * 1024 * time));

    for (int i = 0; i < blocksPerGrid; ++i ) {
      if(out[i] != expectedResult) {
        printf("diff found, i=%d, out[i]=%d\n", i, out[i]);
      }
    }
  }

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
  return 0;
}
