#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdlib>
using namespace std;

int main(int argc, char** argv) {

  const int N = 256;
  const int M = 64;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  vector<float> subA(N*N/M);
  vector<float> subB(N*N/M);
  vector<float> subC(N*N/M, 0);

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }

  int offset = N/M*blockIdx.x;
  for (int i=0; i<N/M; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/M; j++)
      subB[N/M*i+j] = B[N*i+j+offset];

  float *cuA;
  float *cuB;
  float *cuC;

  cudaMalloc((void**)&cuA,N*N/M*sizeof(float));
  cudaMalloc((void**)&cuB,N*N/M*sizeof(float));
  cudaMalloc((void**)&cuC,N*N/M*sizeof(float));
  cudaMemcpy(cuA,subA,N*N/M*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(subB,cuB,N*N/M*sizeof(float),cudaMemcpyHostToDevice);
  //cudaMemcpy(a,b,Bytes,cudaMemcpyHostToDevice);

  //target parallelrithm

  double comp_time = 0, comm_time = 0;
    auto tic = chrono::steady_clock::now();
    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        for (int k=0; k<N; k++)
          C[N*i+j] += A[N*i+k] * B[N*k+j];

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
