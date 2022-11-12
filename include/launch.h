#pragma once
#include"cuda_includes.h"


static cudaError_t error;

#define CUDA_SAFE_CALL(_function_)                        \
error = _function_;                                       \
if(error != cudaSuccess){                                 \
    std::cout << "ERROR: " << cudaGetErrorString(error)   \
    << ", " << __FILE__ << ", " << __LINE__ << std::endl; \
    exit(EXIT_FAILURE);}

#define SYNC_KERNEL(_message_)                                    \
cudaDeviceSynchronize();                                          \
error = cudaGetLastError();                                       \
if(error != cudaSuccess){                                         \
std::cout << "KERNEL " << _message_ << " FAILED " <<std::endl;    \
std::cout << "ERROR: " << cudaGetErrorString(error) << std::endl; \
std::cout << "FILE: " << __FILE__ << std::endl;                   \
std::cout << "LINE: " << __LINE__ << std::endl;                   \
exit(EXIT_FAILURE);}


namespace Error {
    static cudaError_t error;
}

namespace Launch{
    static inline cudaDeviceProp deviceProp;

    static inline dim3 num_blocks;
    static inline dim3 threads_per_block;

    static inline int shared_memory_size = 64*sizeof(int)*32;

    static void allocate_shmem(uint size) {
        shared_memory_size = size;
    }

    static void print_params(){
        std::cout << std::endl << "num blocks: " << num_blocks.x << ", " << num_blocks.y << ", " << num_blocks.z << std::endl;
        std::cout << "threads_per_block: " << threads_per_block.x << ", " << threads_per_block.y << ", " << threads_per_block.z << std::endl;
        std::cout << "shmem size: " << shared_memory_size << std::endl << std::endl;
    }

    template<class Func>
    static void calculate_occupancy(Func kernel){
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
        int numBlocksPerSm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor( &numBlocksPerSm, kernel, 1024, Launch::shared_memory_size);
        Launch::num_blocks = {numBlocksPerSm * Launch::deviceProp.multiProcessorCount,1,1};
        Launch::threads_per_block = {1024,1,1};
    }

    static void kernel_2d(int x_dim, int y_dim){
        uint block_dim_x = (x_dim - (x_dim % 32))/32;
        uint block_dim_y = (y_dim - (y_dim % 32))/32;

        Launch::num_blocks = {block_dim_x + 1, block_dim_y + 1, 1};
        Launch::threads_per_block = {32,32,1};
    }
    
}

//kernel macros

#define GET_DIMS(_first_, _second_)\
const int _first_ = (blockIdx.x * blockDim.x) + threadIdx.x;\
const int _second_ = (blockIdx.y * blockDim.y) + threadIdx.y;\
const int& _FIRST_ = _first_;\
const int& _SECOND_ = _second_;

#define CHECK_BOUNDS(_first_, _second_)\
if((_FIRST_ >= _first_)||(_SECOND_ >= _second_)){return;}