#pragma once

#include "cuda_includes.h"
#include "standard_includes.h"

#include "../core/structs/structs.h"
#include "macros.h"

#include "launch.h"

// Transpose a Matrix array
__global__ static void Transpose(Device_Matrix input, Device_Matrix output) {

  GET_LANEID;
  GET_WARPID;

  // Construct output matrix
  // Matrix output = MatrixData(input.bit_height(), input.bit_width());

  // Get byte dimensions
  const int width_bytes = STEP128(input.bit_width()) * 8;
  const int height_bytes = STEP8(input.bit_height());

  // Create unique thread id (tid) and iterate
  for (int tid = blockIdx.x * 1024 + warpid * 32 + laneid;
       tid < height_bytes * width_bytes; tid += gridDim.x * 1024) {
    // Give each chunk unique coordinate
    const int ex = tid % width_bytes;
    const int ey = tid / width_bytes;

    // Load appropriate 8 x 8 per thread and transpose in place
    Chunk chunk = input.get_chunk(ex, ey);
    chunk.transpose();

    // Set chunk to transposed chunk location
    output.set_chunk(ey, ex, chunk);
  }
}

//! NOTE: This NOT's to the most appropriate BYTE additional care must be taken
//! when combining this with other bit arithmetic as it may lead to undefined
//! behaviour - eg. XOR(NOT(X),X) will lead to 1's in undefined bit space!!!
__global__ static void NOT(Device_Matrix input) {
  GET_DIMS(x, y);
  CHECK_BOUNDS(input.element_dims[0], input.element_dims[1]);

  input(x, y) = ~input(x, y);
}

__global__ static void AND(Device_Matrix LHS, Device_Matrix RHS) {
  GET_DIMS(x, y);
  CHECK_BOUNDS(LHS.element_dims[0], LHS.element_dims[1]);

  LHS(x, y) = LHS(x, y) & RHS(x, y);
}

__device__ static uchar *rowXOR(uchar *LHS, uchar *RHS) {

  if (sizeof(LHS) != sizeof(RHS)) {
    printf("Size of LHS != Size of RHS !!!");
    return;
  }

  uchar out[sizeof(LHS)];

  for (int i = 0; i < sizeof(LHS); i++) {
    out[i] = LHS[i] ^ RHS[i];
  }

  return out;
}