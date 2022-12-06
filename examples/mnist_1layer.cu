#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>

#include "util.cuh"
#include "layer.h"

#include "data_utils.h"


using namespace std;




int main()
{    
    int dev = 0;
    cudaSetDevice(dev);
    

    // =============== Get Data and Label =================
    uchar * train_data = ReadMNISTImages(".data/mnist/train-images.idx3-ubyte");
    uchar * train_labels = ReadMNISTLabels(".data/minst/train-labels.idx1-ubyte");
    uchar * test_data = ReadMNISTImages(".data/mnist/t10k-images.idx3-ubyte");
    uchar * test_labels = ReadMNISTLabels(".data/minst/t10k-labels.idx1-ubyte");

    // ================= Set Network =================

    CUDA_SAFE_CALL(cudaGetDeviceProperties(&Launch::deviceProp, 0));

    FcLayer layer1 = FcLayer(MNIST_IMAGE_SIZE,10,BATCH);

    // Initialize parameters
    layer1.weights.fill_random();
    layer1.biases.fill(128);
    layer1.weight_counters.fill();
    layer1.bias_counters.fill();

    // Loading data caches
    uchar batch_slice[BATCH*MNIST_IMAGE_SIZE];
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
            batch_slice[row*MNIST_IMAGE_SIZE+col] = train_data[(row+i*BATCH)*MNIST_IMAGE_SIZE+col];

            // Specific loop for one-hot encoding.
            if (j == 0) {
                for (int k = 0; k < MNIST_NUM_CLASSES; k++) {
                    if (k == train_labels[row+i*BATCH]) {
                        label_slice[row*MNIST_NUM_CLASSES+k] = 1;
                    }
                }
            }
        }

        // Pack data into layer structures
        layer1.input = PackHostMatrix(batch_slice,MNIST_IMAGE_SIZE,BATCH,MNIST_DATA_THRESHOLD);
        for (int j = 0; j < BATCH*MNIST_IMAGE_SIZE; j++) {
            bool deal = layer1.input.host_data[j] == PackHostMatrix(batch_slice,MNIST_IMAGE_SIZE,BATCH,MNIST_DATA_THRESHOLD).host_data[j];
            uchar val1 = layer1.input.host_data[j];
            uchar val2 = PackHostMatrix(batch_slice,MNIST_IMAGE_SIZE,BATCH,MNIST_DATA_THRESHOLD).host_data[j];
            bool deal2 = val1 == val2;
            printf("%s","hello world");
        }
        layer1.output_label = PackHostMatrix(label_slice,10,BATCH,0);

        // Train network
        layer1.forward();
        layer1.back();
    }

    // ================= Test Network =================
    
    std::cout << "Testing Network..." << std::endl;

    return 0;
}