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

  // Block cache memory
  extern __shared__ int Cs[64];

  // For later use
  typedef cub::BlockReduce<int, 1024> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // bn short for batch number
  for (int bn = 0; bn < batch_blocks; bn++) {

    // gwid short for global warp ID
    for (int gwid = blockIdx.x * 32 + warpid;
         gwid < input_blocks * output_blocks; gwid += gridDim.x * 32) {

      // Compute block coordinates
      const int bx = gwid % output_blocks; // which block column
      const int by = gwid / output_blocks; // which block row

      // Matrix fragments and accumulators
      fragment<matrix_a, 8, 8, 128, b1, row_major> A_frag;
      fragment<matrix_b, 8, 8, 128, b1, col_major> B_frag;
      fragment<accumulator, 8, 8, 128, int> C_frag;

      // Zero-out accumulator fragment
      fill_fragment(C_frag, 0);

      // Keep track of number of thread blocks needed in terms of block threads
      // and warps
      int num_input_thread_blocks = STEP32(input_blocks);
      int num_input_warps = STEP32(input_blocks) * 32;

      // Warp to block mapping
      int bi = gwid % num_input_warps; // Which input block
      int bw = gwid / num_input_warps; // Which weight block

      int Ds[64];

      if (bi < input_blocks) {

        // One 8*128 bit matrix multiply-accumulate per warp
        load_matrix_sync(A_frag, &input(bi, bn, 0, 0), 128);
        load_matrix_sync(B_frag, &weights(bi, bw, 0, 0), 128);
        bmma_sync(C_frag, A_frag, B_frag, C_frag);

        __syncthreads();

        store_matrix_sync(Ds, C_frag, 8, mem_row_major);
      }

      __syncthreads();

// Sum computations into global memory
#pragma unroll
      for (int i = 0; i < 64; i++) {
        Cs[i] = BlockReduce(temp_storage).Sum(Ds[i]);
        Cs[i] /= 32;
        layer_cache[i + (blockIdx.x / num_input_thread_blocks) * 64] += Cs[i];
        Cs[i] = 0;
      }

      __syncthreads();
      // gtid = Global thread ID
      int gtid = laneid + gwid * 32;
      if (gtid < 64 * output_blocks) {
        // Load activations into threads
        int activation = layer_cache[gtid];

        // And reuse this piece of code with a few tweaks...

        // We now map threads to upper 4-by-8 half of the output 8-by-8 in a
        // row-major format
        const int gy = laneid % 8;
        const int gx = laneid / 8;

        // We check if an output int should exist first in the upper half
        bool v0 = ((by * 8 + gy) < output_bits) && ((bx * 8 + gx) < batch_bits);

        // And then we check again in the lower half of the 8-by-8
        bool v1 =
            ((by * 8 + gy) < output_bits) && ((bx * 8 + gx + 4) < batch_bits);

        // Is the int greater than the corresponding output bias?
        v0 &= (Cs[warpid * 64 + laneid] >= biases[by * 8 + gy]);

        // Check again for the lower half of the 8-by-8 int array
        v1 &= (Cs[warpid * 64 + 32 + laneid] >= biases[by * 8 + gy]);

        // Some elaborate bit hack to pack our results into 2 "half chunks" of 4
        // bytes
        union {
          uin32 data;
          uin8 elements[4];
        } p0, p1;
        p0.data = __brev(__ballot_sync(0xFFFFFFFF, v0));
        p1.data = __brev(__ballot_sync(0xFFFFFFFF, v1));
        __syncthreads();

        // Here we deviate slightly from prediction mode and begin to compute
        // error statistics using the output_label matrix.
        Chunk truth = output_label.get_chunk(by, bx);
        Chunk out = Chunk(p0.data, p1.data);

        // generate FP/FN chunks
        Chunk FP = ChunkAND(ChunkNOT(truth), out);
        Chunk FN = ChunkAND(truth, ChunkNOT(out));

        // We are now syncing across warps and should treat our data accordingly
        // each warp at this stage is working on 8 columns of weight matrix

        // Iterate over byte rows in Chunk
        for (int i = 0; i < 8; i++) {

          // Iterate over length of weights bit matrix
          for (int j = 0; j < weights.bit_height(); j++) {

            Chunk bad_fp_weights = Byte2Chunk(FP.data[i] & weights(bx, j));
            Chunk bad_fn_weights = Byte2Chunk(FN.data[i] & weights(bx, j));

            // Iterate through columns of row of bit matrix
            for (int k = 0; k < 8; k++) {

              // If its a bad weight - add to its counter
              if ((bool)bad_fp_weights.data[k] || (bool)bad_fn_weights[k]) {

                weight_counters[8 * gwid + j * output_bits + k] += 1;
                bias_counters[8 * gwid + k] +=
                    bad_fp_weights.data[k] - bad_fn_weights[k];

                if (weight_counters[8 * gwid + j * output_bits + k] %
                        weight_threshold ==
                    0) {
                  // Reset relevant counter, flip relevant weight
                  weight_counters[8 * gwid + j * output_bits + k] = 0;
                  weights(bx, j) ^= 0x80 >> k;

                } else if (bias_counters[8 * gwid + k] % bias_threshold == 0) {
                  // Increment or decrement depending on signal
                  if (bias_counters[8 * gwid + k] > 0) {
                    biases[8 * gwid + k] += 1;
                  } else {
                    biases[8 * gwid + k] -= 1;
                  }

                  // Reset relevant bias counter
                  bias_counters[8 * gwid + k] = 0;
                }
              }
            }
          }
        }

        // Iterate over length of weights bit matrix
        for (int j = 0; j < weights.bit_height(); j++) {
          // Iterate through columns of row of bit matrix
          for (int k = 0; k < 8; k++) {
            // Decrement all weight counters
            weight_counters[8 * gwid + j * output_bits + k] -= 1;
            // Move all bias counters closer to zero
            bias_counters[8 * gwid + k] +=
                (bias_counters[8 * gwid + k] > 0 ? -1 : 1);
          }
        }
        // Use the simple matrix function to set the half chunks
        output.set_half_chunks(by, bx, p0.elements, p1.elements);
      }
    }

    layer_cache[64 * output_blocks] = {0};
  }
}

#pragma endregion

#pragma region CvLayer

__global__ static void CvLayerBk(Device_Matrix input) {
  using namespace nvcuda;
  using namespace nvcuda::wmma::experimental;

  GET_WARPID;
  GET_LANEID;
}

#pragma endregion CvLayer