#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

#include "data_utils.h"
#include "layer.h"
#include "mnist/mnist_reader.hpp"
#include "util.cuh"

int main() {
  int dev = 0;
  cudaSetDevice(dev);

  // =============== Get Data =================
  std::cout << "Loading Data..." << std::endl;
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
          "./.data/mnist");

  // ================= Set Network =================

  CUDA_SAFE_CALL(cudaGetDeviceProperties(&Launch::deviceProp, 0));

  FcLayer layer1 = FcLayer(MNIST_IMAGE_SIZE, MNIST_NUM_CLASSES, BATCH);

  // Initialize parameters
  layer1.weights.fill_random();
  layer1.biases.fill(MNIST_IMAGE_SIZE / 2);

  // Loading raw uint8_t MNIST data in caches
  uchar batch_slice[BATCH * MNIST_IMAGE_SIZE] = {0};
  uchar label_slice[BATCH * MNIST_NUM_CLASSES] = {0};
  Host_Matrix train_batch;
  Host_Matrix labels;

  // ================= Train Network =================

  std::cout << "Training Network..." << std::endl;
  // Get batched data into single array
  for (int i = 0; i < MNIST_DATA_LENGTH / BATCH; i++) {
    for (int j = 0; j < BATCH * MNIST_IMAGE_SIZE; j++) {

      // Rows and columns across the MNIST matrix
      int row = j / MNIST_IMAGE_SIZE;
      int col = j % MNIST_IMAGE_SIZE;

      // Load correct data for batch slice.
      batch_slice[row * MNIST_IMAGE_SIZE + col] =
          dataset.training_images[row + i][col];

      // Specific loop for one-hot encoding.
      if (col == 0) {
        for (int k = 0; k < MNIST_NUM_CLASSES; k++) {
          if (k == dataset.training_labels[row + i]) {
            label_slice[row * MNIST_NUM_CLASSES + k] = 1;
          }
        }
      }
    }

    // Pack data into layer structures
    layer1.input = PackHostMatrix(batch_slice, MNIST_IMAGE_SIZE, BATCH,
                                  MNIST_DATA_THRESHOLD);
    layer1.output_label =
        PackHostMatrix(label_slice, MNIST_NUM_CLASSES, BATCH, 0);

    // Train network
    layer1.predict();
    layer1.train();
  }

  // ================= Test Network =================

  std::cout << "Testing Network..." << std::endl;

  return 0;
}