#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdlib>
using namespace std;


__global__ void mtrix(float *a, float *b, float*c,int N){
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        c[N*i+j] += a[N*i+k] * b[N*k+j];

}
int main(int argc, char** argv) {

  const int N = 256;
  const int M = 64;
  float A[N*N];
  float B[N*N];
  float C[N*N];

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }


  float *cuA;
  float *cuB;
  float *cuC;

  cudaMalloc((void**)&cuA,N*N*sizeof(float));
  cudaMalloc((void**)&cuB,N*N*sizeof(float));
  cudaMalloc((void**)&cuC,N*N*sizeof(float));
  cudaMemcpy(cuA,A,N*N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(cuB,B,N*N*sizeof(float),cudaMemcpyHostToDevice);
  //cudaMemcpy(a,b,Bytes,cudaMemcpyHostToDevice);

  //target parallelrithm

  double comp_time = 0, comm_time = 0;
    auto tic = chrono::steady_clock::now();
    mtrix<<<(N*N+M-1)/M,M>>>(cuA,cuB,cuC,N);
    cudaDeviceSynchronize();
    cudaMemcpy(cuC,C,N*N*sizeof(float),cudaMemcpyDeviceToHost);
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();


  // targets end here
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);

    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);

    cudaFree(cuA);
    cudaFree(cuB);
    cudaFree(cuC);

}
