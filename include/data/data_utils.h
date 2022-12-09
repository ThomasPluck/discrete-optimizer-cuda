#include <iostream>
#include <vector>
#include <fstream>
#include "structs.h"

#define BATCH 32

#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_SIZE 784
#define MNIST_DATA_LENGTH 60000
#define MNIST_DATA_THRESHOLD 50
#define MNIST_NUM_CLASSES 10

Host_Matrix PackHostMatrix(uchar * DataToPack, int bit_width, int bit_height, int byte_threshold) {

    Host_Matrix output(bit_width,bit_height);

    uint8_t bits = 0;

    // Binarize the matrix in row major format
    // Iterate first over rows and then over columns
    for (int i = 0; i < PAD8(bit_height); i++) {
    for (int j = 0, k = 0; j < PAD128(bit_width); j++, k++) {

        k %= 8;
        // If you've reached of the byte and you're within in bounds, commit it.
        if (k == 0 && j != 0) {
            int x_coord = (STEP8(j % PAD128(bit_width))) - 1;
            output(x_coord,i) = bits;
            bits = 0;
        }

        // If you're within bounds, binarize and add to final data output.
        if (i < bit_height && j < bit_width) {
            bits += (DataToPack[i*bit_width+j] > byte_threshold ? (uchar) 128 : (uchar) 0) >> k;
        } else {
            continue;
        }
        
    }}

    output.upload();

    return output;
}