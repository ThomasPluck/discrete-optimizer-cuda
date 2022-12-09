#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>

#include "util.cuh"
#include "layer.h"

#include "data_utils.h"
#include "mnist/mnist_reader.hpp"

int main()
{    
    int dev = 0;
    cudaSetDevice(dev);
    

    // =============== Get Data =================
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(".data/mnist");

    // ================= Set Network =================

    CUDA_SAFE_CALL(cudaGetDeviceProperties(&Launch::deviceProp, 0));

    FcLayer layer1 = FcLayer(MNIST_IMAGE_SIZE,10,BATCH);

    // Initialize parameters
    layer1.weights.fill_random();
    layer1.biases.fill(128);

    // Loading data caches
    uchar batch_slice[BATCH*MNIST_IMAGE_SIZE] = {0};
    uchar label_slice[BATCH*MNIST_NUM_CLASSES] = {0};
    Host_Matrix train_batch;
    Host_Matrix labels;
    
    // ================= Train Network =================

    std::cout << "Training Network..." << std::endl;
    for(int i = 0; i < MNIST_DATA_LENGTH/BATCH; i++){

        // Get batched data into single array
        for(int j = 0; j < BATCH*MNIST_IMAGE_SIZE; j++) {

            // Rows and columns across the MNIST matrix
            int row = j / MNIST_IMAGE_SIZE;
            int col = j % MNIST_IMAGE_SIZE;

            // Load correct data for batch slice.
            batch_slice[row*MNIST_IMAGE_SIZE+col] = mnist.training_images[(row+i*BATCH)*MNIST_IMAGE_SIZE+col];

            // Specific loop for one-hot encoding.
            if (j == 0) {
                for (int k = 0; k < MNIST_NUM_CLASSES; k++) {
                    if (k == mnist.training_labels[row+i*BATCH]) {
                        label_slice[row*MNIST_NUM_CLASSES+k] = 1;
                    }
                }
            }
        }

        // Pack data into layer structures
        layer1.input = PackHostMatrix(batch_slice,MNIST_IMAGE_SIZE,BATCH,MNIST_DATA_THRESHOLD);
        layer1.output_label = PackHostMatrix(label_slice,MNIST_NUM_CLASSES,BATCH,0);

        // Train network
        layer1.forward();
        layer1.back();
    }

    // ================= Test Network =================
    
    std::cout << "Testing Network..." << std::endl;

    return 0;
}