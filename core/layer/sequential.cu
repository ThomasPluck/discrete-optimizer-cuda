#include "sequential.h"
#include "util.cuh"

#pragma region Sequential

Sequential::Sequential(std::vector<Layer> _layers, int _batch_size) {
  std::cout << "Running Sequential Model Constructor..." << std::endl;

  // Check that relevant parameters match
  for (int i = 0; i < _layers.size(); i++) {
    if (_layers[i].batch_bits != _batch_size) {
      throw std::range_error("Sequential and layer batch size do not match");
    }
  }

  // Copy
  layers = _layers;
  
  // Identify key host matrices with relevant layer matrices
  Host_Matrix &input = layers[0].input;
  Host_Matrix &output = layers[layers.size() - 1].output;
  Host_Matrix &output_label = layers[layers.size() - 1].output_label;

  // Link layers together
  for (int i = 1; i < layers.size(); i++) {
    link(&layers[i - 1], &layers[i]);
  }

  // Add some useful parameters
  batch_bits = _batch_size;
  batch_blocks = STEP8(_batch_size);
}

void Sequential::predict() {
  for (int i = 0; i < layers.size(); i++) {
    layers[i].predict();
  }
}

#pragma endregion