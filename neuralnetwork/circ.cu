// nvcc circ.cu -lcufft -lcublas

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <assert.h>

#include <cufft.h>
#include <cublas_v2.h>

using namespace std;

struct GenRand
{
    __device__
    float operator () (int idx)
    {
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist;
        randEng.discard(idx);
        return uniDist(randEng);
    }
};

template <typename Dtype>
struct GenCirculant 
{
    Dtype *r_, *m_;
    int N_;
    bool tran_;
    GenCirculant(Dtype *r, Dtype *m, int N, bool tran) : 
      r_(r), m_(m), N_(N), tran_(tran) {}
    __device__
    Dtype operator () (int idx)
    {
      Dtype val;
      if (tran_) {
        val = r_[idx];
      } else {
        val = r_[idx ? (N_-idx) : idx];
      }

      for (int i = 0; i < N_; ++i) {
         m_[i * N_ + (idx+i)%N_] = val;
      }
      return val;
    } 
};

template <typename Dtype>
void get_circulant(thrust::device_vector<Dtype>& r, 
                   thrust::device_vector<Dtype>& m,
                   bool tran)
{
  const int N = r.size();
  assert(m.size() == N * N);
  
  thrust::transform(
     thrust::make_counting_iterator(0),
     thrust::make_counting_iterator(N),
     r.begin(),
     GenCirculant<Dtype>(thrust::raw_pointer_cast(&r[0]), 
                  thrust::raw_pointer_cast(&m[0]), 
                  N,
                  tran));    
}

void get_fft(cufftReal *d_in_data, 
             cufftComplex *data,
             int N) // N for d_in_data
{
    int nSamples = N;
    int DATASIZE = N; // FFTSIZE
    int batch = nSamples / DATASIZE;

    cufftHandle plan;

    int rank = 1;                           // --- 1D FFTs
    int n[] = { DATASIZE };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = DATASIZE, odist = (DATASIZE / 2) + 1; // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)

    if(cufftPlanMany(&plan, rank, n,
              inembed, istride, idist,
              onembed, ostride, odist, CUFFT_R2C, batch) != CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT error: Plan failed");
      return;
    } 

    if (cufftExecR2C(plan, d_in_data, data) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
      return;
    }

    if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr, "Cuda error: Failed results copy\n");
      return;
    }
}


__global__ void ComplexPointwiseMul(
  cufftComplex *a, 
  cufftComplex *b, 
  cufftComplex *c,
  int size)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < size;
       i += blockDim.x * gridDim.x)
  {
    c[i] = cuCmulf(a[i], b[i]);
  }
}

void pointwise_mul(cufftComplex *d1,
                   cufftComplex *d2,
                   cufftComplex *d,
                   int N)
{
  ComplexPointwiseMul<<<(N + 1023) / 1024, 1024>>>(d1, d2, d, N);
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr, "Cuda error: Failed results copy\n");
      return;
  }
}



void get_ifft(cufftComplex *d_in_data, 
              cufftReal *data,
              int N) // N for data
{
    int nSamples = N;
    int DATASIZE = N; // FFTSIZE
    int batch = nSamples / DATASIZE;

    cufftHandle plan;

    int rank = 1;                           // --- 1D FFTs
    int n[] = { DATASIZE };                 // --- Size of the Fourier transform
    int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
    int idist = (DATASIZE / 2) + 1, odist = DATASIZE; // --- Distance between batches
    int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
    int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)

    if(cufftPlanMany(&plan, rank, n,
              inembed, istride, idist,
              onembed, ostride, odist, CUFFT_C2R, batch) != CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT error: Plan failed");
      return;
    } 

    if (cufftExecC2R(plan, d_in_data, data) != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
      return;
    }

    if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr, "Cuda error: Failed results copy\n");
      return;
    }

    // divide by N to get desired value
    // http://stackoverflow.com/questions/25034609/not-the-same-image-after-cuda-fft-and-ifft 
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1. / N;
    cublasSscal(handle, N, &alpha, (float *) data, 1);
    cublasDestroy(handle);
}


// https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
void get_cr(const float *A, 
            const float *B, 
            float *C, 
            const int m, 
            const int k, 
            const int n) {
  // C(m,n) = A(m,k) * B(k,n)
  int lda=m,ldb=k,ldc=m;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  cublasDestroy(handle);
}

float get_diff_norm(const float *A, const float *B, const int N)
{
  thrust::device_vector<float> diff(A, A+N);
  float res = 0;
  const float alpha = -1;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSaxpy(handle, N, &alpha, B, 1, thrust::raw_pointer_cast(&diff[0]), 1);
  cublasSasum(handle, N, thrust::raw_pointer_cast(&diff[0]), 1, &res);
  cublasDestroy(handle);
  return res;
}

template<typename T>
void print_matrix(thrust::device_vector<T>& m, int R, int C)
{
  for (int i = 0; i < R; ++i) {
    thrust::copy(m.begin() + i * C, 
                 m.begin() + (i+1) * C, 
                 std::ostream_iterator<T>(std::cout, " "));
    cout << endl;
  }
}

void circulant_test()
{
  for (int N = 128; N <= 65536; N <<= 1) {
    // https://codeyarns.com/2013/10/31/how-to-generate-random-numbers-in-thrust/https://codeyarns.com/2013/10/31/how-to-generate-random-numbers-in-thrust/
    thrust::device_vector<float> r(N);
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(N),
        r.begin(),
        GenRand());    
     

    thrust::device_vector<float> x(N);
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(N),
        x.begin(),
        GenRand()); 
    
    // http://stackoverflow.com/questions/27324686/cuda-how-do-i-use-float-audio-data-with-cufft
    thrust::device_vector<cufftComplex> fftx(N/2 + 1);
    get_fft((cufftReal *) thrust::raw_pointer_cast(&x[0]),
            thrust::raw_pointer_cast(&fftx[0]),
            N);
    {
      // check fft is right
      thrust::device_vector<float> x2(N);
      get_ifft(thrust::raw_pointer_cast(&fftx[0]),
               (cufftReal *) thrust::raw_pointer_cast(&x2[0]),
               N);
      float diff_norm = get_diff_norm(
                          thrust::raw_pointer_cast(&x[0]),
                          thrust::raw_pointer_cast(&x2[0]),
                          N);
      //print_matrix(x, 1, N);
      //print_matrix(x2, 1, N);
      cout << "Check fft is right, avg(|diff|)=" << diff_norm << endl;
    }

    thrust::device_vector<cufftComplex> fftr(N/2 + 1);
    get_fft((cufftReal *) thrust::raw_pointer_cast(&r[0]),
            thrust::raw_pointer_cast(&fftr[0]),
            N);

    thrust::device_vector<cufftComplex> fftrx(N/2 + 1);
    pointwise_mul(thrust::raw_pointer_cast(&fftx[0]),
                  thrust::raw_pointer_cast(&fftr[0]),
                  thrust::raw_pointer_cast(&fftrx[0]),
                  fftrx.size());
    
    thrust::device_vector<float> result1(N);
    get_ifft(thrust::raw_pointer_cast(&fftrx[0]),
             (cufftReal *) thrust::raw_pointer_cast(&result1[0]),
             N);


    thrust::device_vector<float> m(N * N);
    get_circulant(r, m, true);
    //print_matrix(m, N, N);

    thrust::device_vector<float> result2(N);
    get_cr(thrust::raw_pointer_cast(&m[0]),
           thrust::raw_pointer_cast(&r[0]),
           thrust::raw_pointer_cast(&result2[0]),
           N, N, 1);
    
    float diff_norm = get_diff_norm(
                        thrust::raw_pointer_cast(&result1[0]),
                        thrust::raw_pointer_cast(&result2[0]),
                        N);
    //print_matrix(result1, 1, N);
    //print_matrix(result2, 1, N);
    cout << "N=" << N << ", avg(|diff|)=" << diff_norm/N << endl;
  }
}

int main()
{
  circulant_test();
  return 0;
}
