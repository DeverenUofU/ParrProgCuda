#include <stdio.h>
#include <time.h>
#include <math.h>
#define threshold 0.0001
#define FIXME 1


void checkCUDAError(const char *msg);

const int DSIZE = 1024;
cudaEvent_t start, stop;
float tstart, elapsedTime;

// matrix multiply kernel: C = A * B
__global__ void mmtt_i4(const float *A, const float *B, float *C, int ds) {

    int tx = threadIdx.x;
    int i = (4 * blockIdx.x) * blockDim.x + tx;  //now to change to multiply for increased stride
    int j = blockIdx.y;                 

    if (i < ds && j < ds) {
        float sum0 = 0.0;
        float sum1 = 0.0;
        float sum2 = 0.0;
        float sum3 = 0.0;
        for (int k = 0; k < ds; ++k) {
            sum0 += A[i + ds*k] * B[k + ds*j];
            sum1 += A[(i+blockIdx.x) + ds*k] * B[k + ds*j];
            sum2 += A[(i+blockIdx.x+blockIdx.x) + ds*k] * B[k + ds*j];
            sum3 += A[(i+blockIdx.x+blockIdx.x+blockIdx.x) + ds*k] * B[k + ds*j];
        }
        C[i * ds + j] = sum0;
        C[(i+blockIdx.x) * ds + j] = sum1;
        C[(i+blockIdx.x+blockIdx.x) * ds + j] = sum2;
        C[(i+blockIdx.x+blockIdx.x+blockIdx.x) * ds + j] = sum3;
    }
}

int main(){

  float *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;
  int i,j,k;

  h_A = new float[DSIZE*DSIZE];
  h_B = new float[DSIZE*DSIZE];
  h_C = new float[DSIZE*DSIZE];
  h_Cref = new float[DSIZE*DSIZE];
  for (i = 0; i < DSIZE*DSIZE; i++){
    h_A[i] = rand();
    h_B[i] = rand();
    h_C[i] = 0;
    h_Cref[i] = 0;}

  for (i=0;i<DSIZE;i++)
   for (k=0;k<DSIZE;k++)
    for (j=0;j<DSIZE;j++)
//   h_Cref[i][j] += h_A[k][i]*h_B[j][k];
     h_Cref[i*DSIZE+j] += h_A[k*DSIZE+i]*h_B[j*DSIZE+k];
  
 // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
  checkCUDAError("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy H2D transfer failure");


  dim3 block(256,1);  
//   dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + 4 - 1) / 4); // Adjusted for i-loop unrolling
  dim3 grid(((DSIZE + block.x - 1) / 4) / block.x, DSIZE);    // Adjusted for x dim,  i-loop unrolling
  printf("Matrix size: %d\n", DSIZE);


  for(int trial=0;trial<3;trial++)
  {
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);
   // Launch kernel
   mmtt_i4<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
   checkCUDAError("GPU kernel launch failure");
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsedTime, start,stop);
   cudaDeviceSynchronize();
   // Copy results back to host
   cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
   checkCUDAError("cudaMemcpy D2H");
   for (int i = 0; i < DSIZE*DSIZE; i++) if (fabs((h_C[i]-h_Cref[i])/h_Cref[i])>threshold) {printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", i, h_C[i], h_Cref[i]); return -1;}
   printf("Trial %d: GFLOPS: %.2f\n",trial,2.0e-6*DSIZE*DSIZE*DSIZE/elapsedTime);
  }
  return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

