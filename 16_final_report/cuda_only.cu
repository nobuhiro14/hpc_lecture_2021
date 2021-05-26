#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include<stdlib.h>
using namespace std;

__global__ void matrix(float *a,float *b,float *c,int N, int offset,int size){
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l <N/size){
    for (int i=0; i<N/size; i++)
      for (int k=0; k<N; k++)
          c[N*i+l+offset] += a[N*i+k] * b[N/size*k+l];
      /*
      for (int i=0; i<N/size; i++)
         for (int k=0; k<N; k++)
           for (int j=0; j<N/size; j++)
            subC[N*i+j+offset] += subA[N*i+k] * subB[N/size*k+j];
      */
  }
}

int main(int argc, char** argv) {

  const int N = 32;
  const int M = 32;
  int size = 4,rank = 0;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  vector<float> subA(N*N/size);
  vector<float> subB(N*N/size);
  vector<float> subC(N*N/size, 0);

  float *a;
  float *b;
  float *c;
  cudaMallocManaged(&a, N*N/size*sizeof(float));
  cudaMallocManaged(&b, N*N/size*sizeof(float));
  cudaMallocManaged(&c, N*N/size*sizeof(float));


  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }


  int offset = N/size*rank;
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];

  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      a[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      b[N/size*i+j] = B[N*i+j+offset];

  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;
  // usual computation
  offset = N/size*((rank) % size);
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N/size; j++)
      for (int k=0; k<N; k++)
        subC[N*i+j+offset] += subA[N*i+k] * subB[N/size*k+j];


  //GPU computation
  double comp_time = 0, comm_time = 0;
    auto tic = chrono::steady_clock::now();
    offset = N/size*((rank) % size);

    matrix<<<(N/size+M-1)/M,M>>>(a,b,c,N,offset,size);
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();

    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();


  for (int i = 0;i<N*N/size;i++)
    c[i] -= subC[i];
  double err = 0;
  for (int i = 0;i<N*N/size;i++)
      err += fabs(c[i]);
  if(rank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("size : %d\n",size);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
