#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;

nt main(void) {
  const int N = 2000;
  const int M = 1024;
  float *a;
  vector<float> subA(N);
  for(int i = 0;i<N;i++)
    subA[i] = 1;
//cudaMalloc((void**)&a,N*(float));
//cudaMemcpy(a,subA,N*(float),cudaMemcpyHostToDevice);


  float A(N);
  for(int i = 0;i<N;i++)
    A[i] = 1;

  cudaMalloc((void**)&a,N*(float));
  cudaMemcpy(a,A,N*(float),cudaMemcpyHostToDevice);



  cudaFree(a);
  
}
