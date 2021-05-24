#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include<stdlib.h>
using namespace std;


__global__ void matrix(float *a, float*b, float*c,int offset,int size ,int N){
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N/size; j++)
      for (int k=0; k<N; k++)
        c[N*i+j+offset] += a[N*i+k] * b[N/size*k+j];

}
int main(int argc, char** argv) {


  const int N = 256;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);

  float *a;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }

cudaMallocManaged(&a, N*N*sizeof(float));


for (int i=0; i<N; i++)
  for (int j=0; j<N; j++)
    a[N*i+j] = A[N*i+j];




  double comp_time = 0, comm_time = 0;
  auto tic = chrono::steady_clock::now();
    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        for (int k=0; k<N; k++)
          C[N*i+j] += A[N*i+k] * B[N*k+j];
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();


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

}
