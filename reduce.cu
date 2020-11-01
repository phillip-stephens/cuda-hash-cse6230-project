#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include <omp.h>
#include <cstddef>

#include "utils.h"
#define NUM_TRIALS 1
#define Giga 1e+9
#define BLOCK_SIZE 32

//Taken from: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void reduceBaseline(const float* In, float* Out) {
    int tid = threadIdx.x; //Local thread ID
    int i = blockIdx.x*blockDim.x + tid; //Global Index
    extern __shared__ float Local[];
    Local[tid] = In[i];  //Load into shared mem
    __syncthreads();
    for(int s=1; s<blockDim.x; s*=2) { //Reduce loop
        if (tid% (2*s) == 0) //Is multiple of s(2, 4, 8, ...)
            Local[tid] += Local[tid+ s];
        __syncthreads();
    }
    if (tid== 0) Out[blockIdx.x] = Local[0];
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *local, unsigned int tid) {
    if(blockSize>= 64) local[tid] += local[tid+ 32];
    if(blockSize>= 32) local[tid] += local[tid+ 16];
    if(blockSize>= 16) local[tid] += local[tid+  8];
    if(blockSize>=  8) local[tid] += local[tid+  4];
    if(blockSize>=  4) local[tid] += local[tid+  2];
    if(blockSize>=  4) local[tid] += local[tid+  1];
}
template<unsigned int blockSize>
__global__ void reduce6(float*In, float*Out, unsigned int n) {
    extern __shared__ float local[];
    unsigned int tid= threadIdx.x;
    unsigned int i= blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize= blockSize*2*gridDim.x;
    local[tid] = 0;

    while(i< n) { local[tid] += In[i] + In[i+blockSize]; i+= gridSize; }
    __syncthreads();
    if(blockSize>= 512) { if(tid< 256) { local[tid] += local[tid+ 256]; }
    __syncthreads();}
     if(blockSize>= 256) { if(tid< 128) { local[tid] += local[tid+ 128]; } 
    __syncthreads();}
    if(blockSize>= 128) { if(tid<  64) { local[tid] += local[tid+  64]; } 
    __syncthreads(); }

    if(tid< 32) warpReduce<blockSize>(local, tid);
    if(tid== 0) Out[blockIdx.x] = local[0];
}

//Code Version: 1=Baseline, 2=SeqAddressing, 3=More work per thread
float reduceDevice(float* array, float* result, T N, int codeVersion)
{
    float et = -omp_get_wtime();
    int size = N*sizeof(float);
    // Load float array to the device
    // float* d_array = array;
    float* d_array;
    gpuErrchk(cudaMalloc((void **)&d_array, size));
    gpuErrchk(cudaMemcpy(d_array, array, size, cudaMemcpyHostToDevice));
    // Allocate Output on the device
    float* d_out_array;
    gpuErrchk(cudaMalloc((void **)&d_out_array, size));
    // Setup the execution configuration
    int blockSize = atoi(getenv("block"));
    int shared_mem_size = blockSize* sizeof(float);
    dim3 dimBlock(blockSize);
    dim3 dimGrid(ceil(N/(float)blockSize));

    int iter = N/blockSize;
    if (codeVersion == 1) {

        cudaEvent_t start, stop;
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));
        gpuErrchk(cudaEventRecord(start, 0));
        reduceBaseline<<<dimGrid, dimBlock, shared_mem_size>>>(d_array, d_out_array);
        while (iter>1) {
            d_array = d_out_array;
            reduceBaseline<<<dimGrid, dimBlock, shared_mem_size>>>(d_array, d_out_array);
            iter/=blockSize;
        }
        gpuErrchk(cudaEventRecord(stop, 0));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&et, start, stop));
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
    } else if(codeVersion == 2) {
        cudaEvent_t start, stop;
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));
        gpuErrchk(cudaEventRecord(start, 0));
        reduceSeqAddress<<<dimGrid, dimBlock, shared_mem_size>>>(d_array, d_out_array);
        while (iter>1) {
            d_array = d_out_array;
            reduceSeqAddress<<<dimGrid, dimBlock, shared_mem_size>>>(d_array, d_out_array);
            iter/=blockSize;
        }
        gpuErrchk(cudaEventRecord(stop, 0));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&et, start, stop));
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
    } else {
        cudaEvent_t start, stop;
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));
        gpuErrchk(cudaEventRecord(start, 0));
        const int this_block_size = 512;
        assert(this_block_size==blockSize);
        reduce6<this_block_size><<<dimGrid, dimBlock, shared_mem_size>>>(d_array, d_out_array, (unsigned int) N);
        while (iter>1) {
            d_array = d_out_array;
            reduce6<this_block_size><<<dimGrid, dimBlock, shared_mem_size>>>(d_array, d_out_array, N);
            iter/=blockSize;
        }
        gpuErrchk(cudaEventRecord(stop, 0));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&et, start, stop));
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));

    }
    // Read C from the device
    gpuErrchk(cudaDeviceSynchronize());
    float* h_out_array = (float *)malloc(size);
    gpuErrchk(cudaMemcpy(h_out_array, d_out_array, size, cudaMemcpyDeviceToHost));
    // for(T i = 0;i<N;i++) {
    //     printf("%f ", h_out_array[i]);
    // }
    // printf("\n***********\n");
    *result = h_out_array[0];
    // Free device matrices
    gpuErrchk(cudaFree(d_array));

    // gpuErrchk(cudaFree(d_out_array));
    return et;
}
int main(int argc, char *argv[])
{
    T N = (argc >= 2) ? (T)atol(argv[1]) : 512;
    T seed = (argc >= 3) ? (T)atol(argv[2]) : 1;

    printf("N = %u seed = %u\n", N, seed);
    float* array = (float*) malloc(N*sizeof(float));
    fill_input(N, seed, array);

    //Sequential
    float* temp_array =(float*) malloc(N*sizeof(float));
    memcpy(temp_array, array, N*sizeof(float));
    float* seq_result =(float*) malloc(sizeof(float));
    double seq_time = reduceSeqHost(seq_result, temp_array, N);

    runTest("Baseline", 1, array, N, seq_result);
    runTest("Sequential Addressing", 2, array, N, seq_result);
    runTest("More Work Per Thread", 3, array, N, seq_result);

    printf("Error: %u\n", cudaGetLastError());
    printf("BlockSize = %d\n", atoi(getenv("block")));


    free(array);
    free(temp_array);


    // ALIGNED_FREE(A);
    // ALIGNED_FREE(B);
    // ALIGNED_FREE(C);
    return 0;
}


    
