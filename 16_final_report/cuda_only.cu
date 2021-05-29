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
  int gpusize;

  cudaGetDeviceCount(&gpusize);
  cudaSetDevice(rank % gpusize);
  const int N = 64,M=256;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  float *subA, *subB, *subC,*recv;
  subA = (float *)malloc(N*sizeof(float));
  subB = (float *)malloc(N*sizeof(float));
  subC = (float *)malloc(N*sizeof(float));
  recv = (float *)malloc(N*sizeof(float));

  float *a;
  float *b;
  float *c;
  cudaMalloc(&a, N*N/size*sizeof(float));
  cudaMalloc(&b, N*N/size*sizeof(float));
  cudaMalloc(&c, N*N/size*sizeof(float));
  cudaDeviceEnablePeerAccess(rank%gpusize, 0);

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
  int offset = N/size*rank;

  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];
  cudaMemcpy(subA,a,N*N/size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(subB,b,N*N/size*sizeof(float),cudaMemcpyHostToDevice);

  double comp_time = 0, comm_time = 0;

    auto tic = chrono::steady_clock::now();
    /*
    offset = N/size*((rank+irank) % size);
    for (int i=0; i<N/size; i++)
      for (int j=0; j<N/size; j++)
        for (int k=0; k<N; k++)
          subC[N*i+j+offset] += subA[N*i+k] * subB[N/size*k+j];
    */
    offset = N/size*((rank) % size);
    printf("before matrix\n");

    //matrix<<<(N+M-1)/M,M>>>(a,b,c,N,offset,size);
    //cudaDeviceSynchronize();
    printf("after matrix\n");


    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    //MPI_Barrier(MPI_COMM_WORLD);
    printf("after comm\n");
    //cudaMemcpy(b,subB,N*N/size*sizeof(float),cudaMemcpyHostToDevice);
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();

  //cudaMemcpy(subC,c,N*N/size*sizeof(float),cudaMemcpyDeviceToHost);


  double err = 0;

  if(rank==0) {
    double time = comp_time+comm_time;
    printf("A[10]: %lf\n",A[10]);
    printf("N    : %d\n",N);
    printf("size : %d\n",size);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  free(subA);
  free(subB);
  free(subC);
  free(recv);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
