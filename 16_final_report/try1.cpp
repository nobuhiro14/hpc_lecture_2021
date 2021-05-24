#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <openacc.h>
using namespace std;

int main(int argc, char** argv) {

  const int N = 256;
  const int M = 64;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  vector<float> *subA;
  vector<float> *subB;
  vector<float> *subC;

  cudaMallocManaged(&subA,N*N/M*sizeof(float))
  cudaMallocManaged(&subB,N*N/M*sizeof(float))
  cudaMallocManaged(&subC,N*N/M*sizeof(float))

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
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];
  //target parallelrithm

  double comp_time = 0, comm_time = 0;
    auto tic = chrono::steady_clock::now();
<<<<<<< HEAD
#pragma acc parallel 
{
 #pragma acc loop
=======
>>>>>>> e6b8ac1d7e33820c8b6ad6a9f79547a56d1d001e
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

    cudaFree(subA);
    cudaFree(subB);
    cudaFree(subC);

}
