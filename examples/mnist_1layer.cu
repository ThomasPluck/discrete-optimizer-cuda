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
    const unsigned batch = 32;
    const unsigned image_height = 28;
    const unsigned image_width = 28;
    const unsigned data_length = 60000;

    const unsigned data_threshold = 50;

    // =============== Get Input and Label =================
    uchar ** ar = ReadMNIST(".data/mnist/train-images.idx3-ubyte");

    // ================= Set Network =================

    CUDA_SAFE_CALL(cudaGetDeviceProperties(&Launch::deviceProp, 0));

    FcLayer layer1 = FcLayer(image_height*image_width,10,batch);

    layer1.weights.fill_random();
    layer1.biases.fill(128);
    layer1.weight_counters.fill();
    layer1.bias_counters.fill();

    uchar ** batch_slice;
    batch_slice = new uchar*[batch];
    for (int i = 0; i < image_height * image_width; i++) {
        batch_slice[i] = new uchar[image_height * image_width];
    }

    // ================= Train Network =================

    std::cout << "Training Network..." << std::endl;
    for(int i = 0; i < data_length/batch; i++){

        // Get batched data into 2D array
        for(int j = 0; j < batch*image_height*image_width; j++) {
            int row = j % (image_height*image_width);
            int col = j / (image_height*image_width);
            batch_slice[row][col] = ar[row+i*batch][col];
        }

        Host_Matrix train_batch = ThresholdAndPack(batch_slice,data_threshold);

        layer1.input = train_batch;
        layer1.input.upload();

        layer1.forward();
        layer1.back();
    }
    
    return 0;
}