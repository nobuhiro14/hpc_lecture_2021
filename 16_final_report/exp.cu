
#include <cstdio>
#include <cmath>
#include <vector>


int main(void) {
  const int N = 2000;
  const int M = 1024;
  float *a;
  vector<float> subA(N);
  for(int i = 0;i<N;i++)
    subA[i] = 1;
cudaMalloc((void**)&a,N*sizeof(float));
cudaMemcpy(a,subA,N*sizeof(float),cudaMemcpyHostToDevice);


  float A[N];
  for(int i = 0;i<N;i++)
    A[i] = 1;

  cudaMalloc((void**)&a,N*sizeof(float));
  cudaMemcpy(a,A,N*sizeof(float),cudaMemcpyHostToDevice);



  cudaFree(a);
  
}
