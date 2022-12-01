#pragma once

#include"standard_includes.h"
#include"cuda_includes.h"

#include"../structs/structs.h"
#include"macros.h"
#include"util.cuh"

//TODO: double check all logic very closely
//--------fully-connected--------

__global__ static void FcLayerFwd(Device_Matrix input, Device_Matrix output, Device_Matrix weights, 
                            Device_Data<uchar> biases, Device_Data<uchar> bias_counters, 
                                Device_Data<uchar> weight_counters,
                                    int input_blocks, int output_blocks, int batch_blocks,
                                    int input_bits, int output_bits, int batch_bits)
{

    using namespace nvcuda;
    using namespace nvcuda::wmma;
    using namespace nvcuda::wmma::experimental;
    using namespace nvcuda::wmma::experimental::precision;

    GET_LANEID;
    GET_WARPID;

    extern __shared__ int Cs[];

    // gwid short for global warp ID
    for (int gwid = blockIdx.x * 32 + warpid;
         gwid < batch_blocks * output_blocks;
         gwid += gridDim.x * 32)
    {
        // Matrix fragments and accumulators
        fragment<matrix_a, 8, 8, 128, b1, row_major> A_frag;
        fragment<matrix_b, 8, 8, 128, b1, col_major> B_frag;
        fragment<accumulator, 8, 8, 128, int> C_frag;

        fill_fragment(C_frag, 0);

        // Compute block coordinates
        const int bx = gwid % output_blocks; // which block column
        const int by = gwid / output_blocks; // which block row
        
        // Loop over block rows/columns
        for (int i=0; i < input_blocks; i++){
            load_matrix_sync(A_frag, &input(i,by,0,0), 128);
            load_matrix_sync(B_frag, &weights(i,bx,0,0), 128);
            bmma_sync(C_frag, A_frag, B_frag, C_frag);
        }
        
        // Matmul output is an 8-by-8 int array per warp
        store_matrix_sync(&Cs[warpid*64], C_frag, 8, mem_row_major);

        // We now map threads to upper 4-by-8 half of the output 8-by-8 in a row-major format
        const int gy = laneid % 8;
        const int gx = laneid / 8;

        // We check if an output int should exist first in the upper half
        bool v0 = ((by*8+gy) < output_bits) && ((bx*8+gx) < batch_bits);

        // And then we check again in the lower half of the 8-by-8
        bool v1 = ((by*8+gy) < output_bits) && ((bx*8+gx+4) < batch_bits);

        // Is the int greater than the corresponding output bias?
        v0 &= (Cs[warpid*64+laneid] >= biases[by*8+gy]);

        // Check again for the lower half of the 8-by-8 int array
        v1 &= (Cs[warpid*64+32+laneid] >= biases[by*8+gy]);

        // Some elaborate bit hack to pack our results into 2 "half chunks" of 4 bytes
        uchar p0[4];
        uchar p1[4];
        p0[0] = __brev(__ballot_sync(0xFFFFFFFF, v0));
        p1[0] = __brev(__ballot_sync(0xFFFFFFFF, v1));
        __syncthreads();

        // Use the simple matrix function to set the half chunks
        output.set_half_chunks(by, bx, p0, p1);
    }
}

#if __CUDACC__ >= 800 // Window of CUDA architectures that support bmmaBitOpAnd

__global__ static void FcLayerBkWeight(Device_Matrix weights, Device_Data<uchar> weight_counters,
                                    Device_Matrix fp_error_t, Device_Matrix fn_error_t,
                                        Device_Matrix input_t, Device_Matrix not_input_t)
{

    GET_WARPID;
    GET_LANEID;

    using namespace nvcuda;
    using namespace nvcuda::wmma;
    using namespace nvcuda::wmma::experimental;
    using namespace nvcuda::wmma::experimental::precision;

    extern __shared__ int bmma_result[];

    // Initialize input

    
    Device_Matrix* I[2] = {&input_t, &not_input_t};
    Device_Matrix* E[2] = {&fp_error_t, &fn_error_t};

    // GWID == Global Warp ID - magic number 32 warps in a block
    for (int gwid = blockIdx.x * warpSize + warpid;
        gwid < input_t.num_blocks_height() * fp_error_t.num_blocks_height();
        gwid+=gridDim.x * warpSize)
    {

        // Matrix fragments and accumulators
        fragment<matrix_a, 8, 8, 128, b1, row_major> A_frag;
        fragment<matrix_b, 8, 8, 128, b1, col_major> B_frag;
        fragment<accumulator, 8, 8, 128, int> C_frag;

        fill_fragment(C_frag, 0);

        // GWID --> Warp's Block Coordinates
        const int by = gwid % fp_error_t.num_blocks_width(); // block column
        const int bx = gwid / fp_error_t.num_blocks_width(); // block row
           
        // A = BMMA_AND(Input^T,FP)
        // B = BMMA_AND(NOT(Input)^T,FP)
        // C = BMMA_AND(Input^T,FN)
        // D = BMMA_AND(NOT(Input)^T,FN)

        // A &= NOT(Weights)
        // B &= Weights
        // C &= Weights
        // D &= NOT(Weights)

        // Blame = A+B+C+D

        for (int j=0; j<4; j++)
        {

            for (int i=0; i < input_t.num_blocks_width(); i++)
            {
                // Always load matrix fragments and accumulate
                load_matrix_sync(A_frag, &(*I[j%2])(i,by,0,0), 128);
                load_matrix_sync(B_frag, &(*E[j/2])(i,bx,0,0), 128);
                bmma_sync(C_frag, A_frag, B_frag, C_frag, bmmaBitOpAND);
            }
            //TODO: double check that weight orientation is correct.
            // Store matrix in shared memory
            store_matrix_sync(&bmma_result[warpid*64], C_frag, 8, mem_row_major);

            Chunk weight_chunk = weights.get_chunk(bx,by);

            // ( j % 3 == 0 ? weight_chunk = ~weight_chunk)
            if (j % 3 == 0) {
                weight_chunk.NOT();
            }

            // Move bit to appropriate test site
            uin32 test_val = 0x80000000 >> laneid;

            // Check for 1 at test_val location in both half matrices and return full or empty uin32s accordingly
            uin32 v0 = (test_val & weight_chunk.halves[0] > 0 ? 0xFFFFFFFF : 0x00000000);
            uin32 v1 = (test_val & weight_chunk.halves[1] > 0 ? 0xFFFFFFFF : 0x00000000);

            // AND appropriately with found values to leave behind appropriate comparisons
            bmma_result[warpid*64+laneid] &= v0;
            bmma_result[warpid*64+laneid+32] &= v1;
            

            // Flip weights if appropriate
            weight_counters[warpid*64+laneid] += bmma_result[warpid*64+laneid];
            weight_counters[warpid*64+laneid] *= -1 * ((weight_counters[warpid*64+laneid] > WEIGHT_THRESHOLD) - 1);

            weight_counters[warpid*64+laneid+32] += bmma_result[warpid*64+laneid+32];
            weight_counters[warpid*64+laneid+32] *= -1 * ((weight_counters[warpid*64+laneid+32] > WEIGHT_THRESHOLD) - 1);

            __syncthreads();

            // Reinitialize C
            fill_fragment(C_frag, 0);

        }

        // TODO: Iterate over properly to ensure all weight counters are decremented.
        // Synchronized counter decrement - determined by user's batch size.
        weight_counters[warpid*64+laneid] -= 1;
        weight_counters[warpid*64+laneid+32] -= 1;
    }
}

#else


// BMMA-free backpropagation
__global__ static void FcLayerBkWeight(Device_Matrix input, Device_Matrix weights, Device_Data<uchar> weight_counters,
                                    Device_Matrix fp_error, Device_Matrix fn_error) {
    
    using namespace nvcuda;

    GET_WARPID;
    GET_LANEID;


    //shmem size is passed in through the kernel launch parameters, inside the kernel we just give the name and 
    //the compiler trusts that we will tell it the size at launch
    //Which neuron caused the error to occur?
    __shared__ int* error_cols;
    //Which sample did this error occur in?
    __shared__ uint* error_rows;

    int MAX_ERRORS = (fp_error.bit_width() * fp_error.bit_height()) / warpSize;

    // GWID == Global Warp ID
    for (int gwid = blockIdx.x * warpSize + warpid;
            gwid < MAX_ERRORS / 8;
                gwid += gridDim.x * warpSize) {

        // Allocate each warp a 32-byte row slice
        int bx = gwid % (block_width * fp_error.num_blocks_width() / warpSize);
        int by = gwid / (block_width * fp_error.num_blocks_width() / warpSize);

        // Retrieve a byte per thread
        uchar fp_error_byte = fp_error(bx+laneid,by);
        uchar fn_error_byte = fn_error(bx+laneid,by);

        // Check each bit to see if they have caused an error
        #pragma unroll
        for (uchar i=0x80, j = 0; i > 0 && j < 8; i >> 1, j++){

            // FP and FN errors are mutually exclusive - so this works

            // Note convention: 0 = no error found, column indexing begins at 1 if FP or -1 if FN
            error_rows[laneid * CHAR_BIT + j] = (i & fp_error_byte > 0 ? bx+laneid+j+1 : 0);
            error_rows[laneid * CHAR_BIT + j] = (i & fn_error_byte > 0 ? -(bx+laneid+j+1) : 0);
            // Same thing, 0 = no error found, column indexing begins at 1
            error_cols[laneid * CHAR_BIT + j] = ((i & fp_error_byte > 0) || (i & fn_error_byte > 0) ? by : 0);

        }

        __syncthreads();

        // Iterate over all members of shared memory
        #pragma unroll
        for (int i=0; i < warpSize * CHAR_BIT; i++) {
            
            // If no error, skip
            if (error_rows[i] == 0 || error_cols[i] == 0) { 
            } else {

                // Take bitwise XOR of input and offending weight 
                uchar * blame = rowXOR(input(error_rows[i]-1),weights(std::abs(error_cols[i])-1));

                //TODO: Find a way to continue iterating until entire blame array is consumed.
                #pragma unroll
                for (int j = 0; j < CHAR_BIT; j++) {

                    // Determine if weight in question is to blame for error
                    bool sign = (error_cols[i] < 0);
                    uchar val = ((blame[laneid] & 0x80 >> j) > 1 ? 1 ^ sign : 0 ^ sign);

                    // Increment weight counter if error is found!
                    weight_counters[laneid + j + weights.bit_width() * error_rows[i]] += val;

                }

            }

            __syncthreads();

        }

    }

    // Synchronized counter decrement - determined by user's batch size.
    weight_counters[warpid*64+laneid] -= 1;
    weight_counters[warpid*64+laneid+32] -= 1;

}

#endif

__global__ static void FcLayerBkBias(Device_Matrix fp_error_t, Device_Matrix fn_error_t, Device_Data<uchar> bias_counters, Device_Data<uchar> biases) {

    GET_WARPID;
    GET_LANEID;

    for (int gwid = blockIdx.x*32+warpid;
        gwid < fp_error_t.num_blocks_width() * fp_error_t.num_blocks_height();
        gwid += gridDim.x*32)
    {

        const int by = gwid % fp_error_t.num_blocks_width(); // block column
        const int bx = gwid / fp_error_t.num_blocks_width(); // block row

        // Get appropriate chunk
        uchar bias_incr[8]; 
        uchar bias_decr[8]; 

        ChunkPOPC(fp_error_t.get_chunk(bx,by), bias_incr);
        ChunkPOPC(fn_error_t.get_chunk(bx,by), bias_decr);

        const int gy = laneid % 8;

        // Only work on bottom 8 lanes per warp
        // TODO: Optimize loop to use warps more densely
        if (laneid < 8) {

            // If bias threshold passed in either direction
            if (abs(bias_counters[by*8 + gy] + bias_incr[gy] - bias_decr[gy]) > BIAS_THRESHOLD) {

                // If bias counters are positive... reset counter and increment bias
                if (bias_counters[by*8 + gy] > 0) {
                    bias_counters[by*8 + gy] = 0;
                    biases[by*8 + gy] += 1;

                // If biases counters are negative... reset counter and decrement bias
                } else {
                    bias_counters[by*8 + gy] = 0;
                    biases[by*8 + gy] -= 1;

                }

            // Else continue building counters
            } else {

                bias_counters[by*8 + gy] += bias_incr[gy];
                bias_counters[by*8 + gy] += bias_decr[gy];
            }
        }
    }
    __syncthreads();
}

//---------convolution--------

__global__ static void CvLayerBk(Device_Matrix input) {

    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;

    GET_WARPID;
    GET_LANEID;


}