#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include<stdlib.h>
using namespace std;

__global__ void matrix(int *a, int*b, int*c, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <=N){

  }
}
int main(int argc, char** argv) {
  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 256;
  const int M = 64;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);
  float subA[N*N/size];
  float subB[N*N/size];
  float subC[N*N/size];
  int gpusize, gpurank ;

  float *a;
  float *b;
  float *c;
  //initial original matrix
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }

//initialize sub matrix for MPI
  int offset = N/size*rank;
  for (int i=0; i<N/size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
  for (int i=0; i<N; i++)
    for (int j=0; j<N/size; j++)
      subB[N/size*i+j] = B[N*i+j+offset];
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;

//start parallerithm
  cudaGetDeviceCount(&gpusize);
  cudaSetDevice(rank % gpusize);
  cudaGetDevice(&gpurank);
  cudaMalloc(&a, N*sizeof(float));
  cudaMalloc(&b, N*sizeof(float));
  cudaMalloc(&c, N*sizeof(float));
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);

cudaMemcpy(a,subA,N*sizeof(float),cudaMemcpyHostToDevice);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
  double comp_time = 0, comm_time = 0;
  for(int irank=0; irank<size; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto tic = chrono::steady_clock::now();
    offset = N/size*((rank+irank) % size);
    for (int i=0; i<N/size; i++)
      for (int j=0; j<N/size; j++)
        for (int k=0; k<N; k++)
          subC[N*i+j+offset] += subA[N*i+k] * subB[N/size*k+j];
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    //cudaMemcpy(c,subC,N*N/size*sizeof(float),cudaMemcpyDeviceToHost);

    MPI_Send(&subB[0], N*N/size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);
    MPI_Recv(&subB[0], N*N/size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
  MPI_Allgather(&subC[0], N*N/size, MPI_FLOAT, &C[0], N*N/size, MPI_FLOAT, MPI_COMM_WORLD);
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  if(rank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
    printf("cuda time: %lf\n",milliseconds);
    printf("cuda est : %lf\n",milliseconds*(N/size)*(N/size));
  }
  cudaFree(a);
  //cudaFree(b);
  //cudaFree(c);
  MPI_Finalize();

}
