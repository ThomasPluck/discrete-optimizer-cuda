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
  extern __shared__ Chunk Es[];

  // brow = batch block row
  for (int brow = 0; brow < batch_blocks; brow ++){

    // GWID = Global Warp ID
    for (int gwid = blockIdx.x * 32 + warpid;
          gwid < PAD32(input_blocks) * output_blocks;
          gwid += gridDim.x * 32) {

      // * Inference step begins here

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

      // * Thread-block sum-reduce

      // Each thread will only require two ints
      int Ds[2] = {0};

      // Final summation will be stored in warp 0's Ds[2]
      for (int i = 0; i < 4; i++) {

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

      // Ensure that bit in question is within bounds
      if (8 * (gwid / 2) + (laneid % 8) < output_bits) {
        // Ds[0] is the final summed output split across threads
        Ds[0] = layer_cache[32 * gwid + laneid];
        // Ds[1] is appropriate bias corresponding to summed output
        Ds[1] = biases[8 * (gwid / 2) + (laneid % 8)];
      }
      
      // Final binary output
      Ds[0] = Ds[0] < Ds[1];

      // Efficiently convert to binarized output
      uchar p0[4], fp[4], fn[4];
      uin32 sync = __brev(__ballot_sync(0xFFFFFFFF, Ds[0]));
      memcpy(p0, &sync, sizeof(uin32));
      __syncthreads();

      // TODO: Implement binarized output

      // * Training portion begins here

      // Write chunks to shared memory
      Es[warpid/2] = output_label.get_chunk(gwid / 2, brow * 8);
      // Create FP/FN vectors at warp-level
      for (int i = 0; i < 4; i++) {
        fp[i] = p0[i] & !Es[gwid/2].data[i+4*(gwid%2)];
        fn[i] = !p0[i] & Es[gwid/2].data[i+4*(gwid%2)];
      }
      
      // * Each warp now has 32 FP/FN bits, or 8 error bytes per warp.

      // global thread ID
      int gtid = laneid % 8 + 8 * (gwid / 2);
      // ! Bit level addressing must be correct to avoid indexing overshoot
      // Iterate along weight columns
      for (int i = 0; i < output_blocks * 16; i++) {

        for (int j = 0; j < 2; j++) {
          // For maximum occupancy, we work through all 4 error bytes with 8 weight column bytes
          // its important to note that each error byte corresponds exactly to 8 weight columns.
          uchar cand = weights(i,gtid);
          
          // Iterate across each bit.
          if (j==0) {
            for (int k = 0; k < 8; k++){
              cand &= (fp[laneid / 8] & (128 >> k)) ? 0xFF : 0x00;
              int loc = k + 8 * (i + gtid);
              if (loc < output_bits * input_bits) {
                weight_counters[loc] += cand;
                bias_counters[loc % output_bits] -= cand;
              }
            }
          // Repeated for the fn segment
          } else {
            for (int k = 0; k < 8; k++){
              cand &= (fn[laneid / 8] & (128 >> k)) ? 0xFF : 0x00;
              int loc = k + 8 * (i + gtid);
              if (loc < output_bits * input_bits) {
                weight_counters[loc] += cand;
                bias_counters[loc % output_bits] += cand;
              }
            }
          }
        }
      }

      // * Begin flipping weights and incrementing/decrementing biases

      // global thread ID
      gtid = laneid + 32 * gwid;

      // Ensure that bit in question is within bounds
      if (gtid < output_bits * input_bits) {
        Ds[0] = weight_counters[gtid];

        // Flip weight if over threshold
        if (Ds[0] > WEIGHT_THRESHOLD) {
          weights(gtid % (8 * output_blocks * 16),
          gtid / (8 * output_blocks * 16)) 
          ^= (Ds[0] > WEIGHT_THRESHOLD)
          << (7 - gtid % 8);
          weight_counters[gtid] = 0;
        }

        // Decrement all weight blame counters
        weight_counters[gtid] -= 1;
      }

      if (gtid < output_bits) {
        Ds[0] = bias_counters[gtid];

        // Increment/decrement biases if over absolute threshold
        if (Ds[0] > BIAS_THRESHOLD) {
          biases[gtid] += 1;
          bias_counters[gtid] = 0;
        } else if (Ds[0] < -BIAS_THRESHOLD) {
          biases[gtid] -= 1;
          bias_counters[gtid] = 0;
        }

        // Increment/decrement counters towards 0.
        int x = bias_counters[gtid];
        bias_counters[gtid] = ((x > 0) - (x < 0)) 
        * (((x > 0) - (x < 0)) 
        * x - 1);
      }

      //TODO: Generate next layer's labels for training.

    }
  }
}