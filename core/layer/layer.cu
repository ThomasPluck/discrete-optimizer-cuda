#include "layer.h"
#include "util.cuh"

#pragma region Layer

Layer::Layer(int _input_size, int _output_size, int _batch_size) {
  std::cout << "Running Layer constructor..." << std::endl;
  std::cout << "Initializing Host_Matrices..." << std::endl;
  input = Host_Matrix(_input_size, _batch_size);
  output = Host_Matrix(_output_size, _batch_size);

  std::cout << "Initializing Host_Data..." << std::endl;
  input_label = Host_Matrix(_input_size, _batch_size);
  output_label = Host_Matrix(_output_size, _batch_size);

  input_blocks = STEP128(_input_size);
  output_blocks = STEP8(_output_size);
  batch_blocks = STEP8(_batch_size);

  input_bits = _input_size;
  output_bits = _output_size;
  batch_bits = _batch_size;
}

Layer::operator Param() { return Param(this); }

Param::Param(Layer *input) {
  input_blocks = input->input_blocks;
  output_blocks = input->output_blocks;
  batch_blocks = input->batch_blocks;

  input_bits = input->input_bits;
  output_bits = input->output_bits;
  batch_bits = input->batch_bits;
}

void link(Layer *prelink, Layer *postlink) {
  if (prelink->output_bits == postlink->input_bits) {
    postlink->input = prelink->output;
    prelink->output_label = postlink->input_label;
  } else {
    throw std::overflow_error("prelink output and postlink input do not match");
  }
}

#pragma endregion

#pragma region FcLayer

FcLayer::FcLayer(int _input_size, int _output_size, int _batch_size)
    : Layer(_input_size, _output_size, _batch_size) {
  std::cout << "Running FcLayer constructor..." << std::endl;

  // Weights and biases
  weights = Host_Matrix(_input_size, _output_size);
  biases = _output_size;

  // Weight and bias counters
  weight_counters = _output_size * _input_size;
  bias_counters = _output_size;

  // Weight and bias threshold paramaters
  weight_threshold = 4;
  bias_threshold = 4;
}

void FcLayer::predict() {
//! Only allow predict if XOR BMMA is not deprecated
#if __CUDACC__ < 900

  // Launch::allocate_shmem(10000);
  Launch::calculate_occupancy(FcLayerPredict);
  Launch::print_params();
  FcLayerPredict<<<Launch::num_blocks, Launch::threads_per_block,
                   Launch::shared_memory_size>>>(
      input, output, weights, biases, input_blocks, output_blocks, batch_blocks,
      input_bits, output_bits, batch_bits);
  SYNC_KERNEL("FcLayerPredict");

#else

  throw std::system_error(
      "bmmaBitOpXOR b1 instructions are deprecated for your card.");

#endif
}

void FcLayer::train() {

  Launch::calculate_occupancy(FcLayerTrain);
  FcLayerTrain<<<Launch::num_blocks, Launch::threads_per_block>>>(
      input, output, weights, output_label, input_label, biases, bias_counters,
      weight_counters, input_blocks, output_blocks, batch_blocks, input_bits,
      output_bits, batch_bits, weight_threshold, bias_threshold);
  SYNC_KERNEL("FcLayerTrain");
}

#pragma endregion

#pragma region CvLayer

// void CvLayer::CvLayer(int _input_size, int _output_size, int _batch_size):
// Layer(_input_size, _output_size, _batch_size){

//     return;

// }

// void CvLayer::predict(){
//     //Launch::calculate_occupancy(CvLayerFwd);
//     //FcLayerPredict<<<Launch::num_blocks,
//     Launch::threads_per_block>>>(this);
//     //SYNC_KERNEL(CvLayerFwd);
// }

// void CvLayer::train(){
//     //Launch::calculate_occupancy(CvLayerBk);
//     //FcLayerTrain<<<Launch::num_blocks, Launch::threads_per_block>>>(this);
//     //SYNC_KERNEL(CvLayerBk);
// }

#pragma endregion