#pragma once

#include "../structs/structs.h"
#include "cuda_includes.h"
#include "macros.h"
#include "standard_includes.h"
#include "util.cuh"

#pragma region FcLayer
__global__ static void FcLayerPredict(Device_Matrix input, Device_Matrix output,
                                      Device_Matrix weights,
                                      Device_Data<uint16_t> biases,
                                      int input_blocks, int output_blocks,
                                      int batch_blocks, int input_bits,
                                      int output_bits, int batch_bits) {

  using namespace nvcuda;
  using namespace nvcuda::wmma;
  using namespace nvcuda::wmma::experimental;
  using namespace nvcuda::wmma::experimental::precision;

  GET_LANEID;
  GET_WARPID;

  extern __shared__ int Cs[];

  // gwid short for global warp ID
  for (int gwid = blockIdx.x * 32 + warpid; gwid < batch_blocks * output_blocks;
       gwid += gridDim.x * 32) {
    // Matrix fragments and accumulators
    fragment<matrix_a, 8, 8, 128, b1, row_major> A_frag;
    fragment<matrix_b, 8, 8, 128, b1, col_major> B_frag;
    fragment<accumulator, 8, 8, 128, int> C_frag;

    fill_fragment(C_frag, 0);

    // Compute block coordinates
    const int bx = gwid % output_blocks; // which block column
    const int by = gwid / output_blocks; // which block row

    // Loop over block rows/columns
    for (int i = 0; i < input_blocks; i++) {
      load_matrix_sync(A_frag, &input(i, by, 0, 0), 128);
      load_matrix_sync(B_frag, &weights(i, bx, 0, 0), 128);
      bmma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    // Matmul output is an 8-by-8 int array per warp
    store_matrix_sync(&Cs[warpid * 64], C_frag, 8, mem_row_major);

    // We now map threads to upper 4-by-8 half of the output 8-by-8 in a
    // row-major format
    const int gy = laneid % 8;
    const int gx = laneid / 8;

    // We check if an output int should exist first in the upper half
    bool v0 = ((by * 8 + gy) < output_bits) && ((bx * 8 + gx) < batch_bits);

    // And then we check again in the lower half of the 8-by-8
    bool v1 = ((by * 8 + gy) < output_bits) && ((bx * 8 + gx + 4) < batch_bits);

    // Is the int greater than the corresponding output bias?
    v0 &= (Cs[warpid * 64 + laneid] >= biases[by * 8 + gy]);

    // Check again for the lower half of the 8-by-8 int array
    v1 &= (Cs[warpid * 64 + 32 + laneid] >= biases[by * 8 + gy]);

    // Some elaborate bit hack to pack our results into 2 "half chunks" of 4
    // bytes
    uchar p0[4];
    uchar p1[4];
    p0[0] = __brev(__ballot_sync(0xFFFFFFFF, v0));
    p1[0] = __brev(__ballot_sync(0xFFFFFFFF, v1));
    __syncthreads();

    // Use the simple matrix function to set the half chunks
    output.set_half_chunks(by, bx, p0, p1);
  }
}

__global__ static void
FcLayerTrain(Device_Matrix input, Device_Matrix output, Device_Matrix weights,
             Device_Matrix output_label, Device_Matrix input_label,
             Device_Data<uint16_t> biases, Device_Data<int> bias_counters,
             Device_Data<int> layer_cache, Device_Data<uchar> weight_counters,
             int input_blocks, int output_blocks, int batch_blocks,
             int input_bits, int output_bits, int batch_bits,
             uchar weight_threshold, uchar bias_threshold) {

  using namespace nvcuda;
  using namespace nvcuda::wmma;
  using namespace nvcuda::wmma::experimental;
  using namespace nvcuda::wmma::experimental::precision;

  GET_LANEID;
  GET_WARPID;

  extern __shared__ int Cs[];
  extern __shared__ uchar Es[];

  // brow = batch block row
  for (int brow = 0; brow < batch_blocks; brow ++){

    // GWID = Global Warp ID
    for (int gwid = blockIdx.x * 32 + warpid;
          gwid < PAD32(input_blocks) * output_blocks;
          gwid += gridDim.x * 32) {

      // Matrix fragments and accumulators
      fragment<matrix_a, 8, 8, 128, b1, row_major> A_frag;
      fragment<matrix_b, 8, 8, 128, b1, col_major> B_frag;
      fragment<accumulator, 8, 8, 128, int> C_frag;

      // Fill accumulator fragment with 0s
      fill_fragment(C_frag, 0);

      // Compute block coordinates
      const int icol = gwid % PAD32(input_blocks); // input column block
      const int wcol = gwid / PAD32(input_blocks); // weight column block

      // Load matrix blocks and compute their product
      if (icol < input_blocks) {
        load_matrix_sync(A_frag, &input(icol, brow, 0, 0), 128);
        load_matrix_sync(B_frag, &weights(icol, wcol, 0, 0), 128);
        bmma_sync(C_frag, A_frag, B_frag, C_frag);
      }

      // Matmul output is an 8-by-8 int array per warp into shared memory
      store_matrix_sync(&Cs[warpid * 64], C_frag, 8, mem_row_major);

      // Each thread will only require two ints
      int Ds[2] = {0};
      // Thread-block sum-reduce
      //! Final summation will be stored in warp 0's Ds[2]
      for (int i = 0; i < log2(32) - 1; i++) {

        if (warpid % (2 << i) == 0) {

          Ds[0] = Cs[64 * warpid + laneid] + Cs[64 * (warpid + (1 << i)) + laneid];
          Ds[1] = Cs[64 * warpid + laneid + 32] + Cs[64 * (warpid + (1 << i)) + laneid + 32];
          Cs[64 * warpid + laneid] = Ds[0];
          Cs[64 * warpid + laneid + 32] = Ds[1];

        }
      }

      // Send to threadblock sum in shared memory to global memory and take final sum
      if (warpid == 0) {
        layer_cache[64 * (blockIdx.x/STEP32(input_blocks)) + laneid] += Ds[0];
        layer_cache[64 * (blockIdx.x/STEP32(input_blocks)) + laneid + 32] += Ds[1];
      }

      // Allocate each thread a unique int and its accompanying bias
      Ds[0] = layer_cache[32 * gwid + laneid];
      Ds[1] = 0;
      if (8 * (gwid / 2) + (laneid % 8) < output_bits) {
        Ds[1] = biases[8 * (gwid / 2) + (laneid % 8)];
      }
      
      // Final binary output
      Ds[0] = Ds[0] < Ds[1];

      // Efficiently convert to binarized output
      uchar p0[4];
      p0[0] = __brev(__ballot_sync(0xFFFFFFFF, Ds[0]));
      __syncthreads();

      // Use the simple matrix function to set the half chunks
      output.set_half_chunks(by, bx, p0, p1);

    }
  }
}