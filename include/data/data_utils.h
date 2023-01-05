#include <fstream>
#include <iostream>
#include <vector>

#include "structs.h"

#define MNIST_BATCH 10000

#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_SIZE 784
#define MNIST_DATA_LENGTH 60000
#define MNIST_DATA_THRESHOLD 50
#define MNIST_NUM_CLASSES 10

Host_Matrix PackHostMatrix(uchar *DataToPack, int bit_width, int bit_height,
                           int byte_threshold) {
  Host_Matrix output(bit_width, bit_height);

  uint8_t bits = 0;

  // Binarize the matrix in row major format
  // Iterate first over rows and then over columns
  for (int i = 0; i < PAD8(bit_height); i++) {
    for (int j = 0, k = 0; j < PAD128(bit_width); j++, k++) {
      k %= 8;
      // If you've reached of the byte and you're within in bounds, commit it.
      if (k == 0 && j != 0) {
        int x_coord = (STEP8(j % PAD128(bit_width))) - 1;
        output(x_coord, i) = bits;
        bits = 0;
      }

      // If you're within bounds, binarize and add to final data output.
      if (i < bit_height && j < bit_width) {
        bits += (DataToPack[i * bit_width + j] > byte_threshold ? (uchar)128
                                                                : (uchar)0) >>
                k;
      } else {
        continue;
      }
    }
  }

  output.upload();

  return output;
}

// Method to unpack host matrix data structure to ordinary boolean array.
bool * UnpackHostMatrix(Host_Matrix DataToUnpack) {

  // Output bool array
  bool output_array[DataToUnpack.bit_height() * DataToUnpack.bit_width()];
  bool temp[8];

  // Iterate across all Host_Matrix bytes
  for (int i = 0; i < 16 * DataToUnpack.block_dims[0]; i++) {
    for (int j = 0; j < 8 * DataToUnpack.block_dims[1]; j++) {

      // If i is a byte known to contain data according to bit width
      if (i < STEP8(DataToUnpack.bit_width())) {

        for (int k = 0; k < 8; k++) {
          temp[k] = (DataToUnpack(i,j) & (128 >> k)) > 0 ? true : false;
        }

        // Iterate across all bits in a byte
        for (int k = 0; k < 8; k++) {
          
          // If selected bit is within bit range.
          if (k + i * 8 < DataToUnpack.bit_width()) {
            output_array[j*DataToUnpack.bit_width()+i] = temp[k];
          }

        }
      }
    }
  }

  return output_array;

}