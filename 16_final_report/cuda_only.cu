#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;

__global__ void matrix(float *a,float *b,float *c,int N, int offset,int size){
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l <N/size){
    for (int i=0; i<N/size; i++)
      for (int j=0; j<N/size; j++)
        c[N*i+j+offset] += a[N*i+l] * b[N/size*l+j];
      /*
      for (int i=0; i<N/size; i++)
         for (int k=0; k<N; k++)
           for (int j=0; j<N/size; j++)
            subC[N*i+j+offset] += subA[N*i+k] * subB[N/size*k+j];
      */
  }
}

int main(int argc, char** argv) {
  int size=1, rank=0;



  const int N = 64;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  float *subA, *subB, *subC,*recv;


  float *a;
  float *b;
  float *c;
  cudaMalloc(a, N*N/size*sizeof(float));
  cudaMalloc(b, N*N/size*sizeof(float));
  cudaMalloc(c, N*N/size*sizeof(float));

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }

/*
  int offset = N/size*rank;
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];
*/
  
  //cudaMemcpy(subA,a,N*N/size*sizeof(float),cudaMemcpyHostToDevice);
  //cudaMemcpy(subB,b,N*N/size*sizeof(float),cudaMemcpyHostToDevice);




  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
