#include "layer.h"
#include "util.cuh"

#pragma region Layer

Layer::Layer(int _input_size, int _output_size, int _batch_size) {
  std::cout << "running Layer constructor..." << std::endl;
  std::cout << "initializing Host_Matrices..." << std::endl;
  input = Host_Matrix(_input_size, _batch_size);
  output = Host_Matrix(_output_size, _batch_size);

  std::cout << "initializing Host_Data..." << std::endl;
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

Param::Param(Layer* input) {
  input_blocks = input->input_blocks;
  output_blocks = input->output_blocks;
  batch_blocks = input->batch_blocks;

  input_bits = input->input_bits;
  output_bits = input->output_bits;
  batch_bits = input->batch_bits;
}

void link(Layer* prelink, Layer* postlink) {
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
  std::cout << "running FcLayer constructor..." << std::endl;

  weights = Host_Matrix(_input_size, _output_size);

  biases = _output_size;
  weight_counters = _output_size * _input_size;
  bias_counters = _output_size;
}

void FcLayer::forward() {
#if __CUDACC__ < 900

  // Launch::allocate_shmem(10000);
  Launch::calculate_occupancy(FcLayerFwd);
  Launch::print_params();
  FcLayerFwd<<<Launch::num_blocks, Launch::threads_per_block,
               Launch::shared_memory_size>>>(
      input, output, weights, biases, weight_counters, bias_counters,
      input_blocks, output_blocks, batch_blocks, input_bits, output_bits,
      batch_bits);
  SYNC_KERNEL("FcLayerFwd");

#else

  throw std::system_error(
      "bmmaBitOpXOR b1 instructions are deprecated for your card.");

#endif
}

void FcLayer::back() {
  // get NOT output_labels
  // get NOT output

  Host_Matrix not_out_label(output.bit_dims[0], output.bit_dims[1]);
  not_out_label.load(output_label.host_data, output.bit_dims[0],
                     output.bit_dims[1]);

  Host_Matrix output_label_matrix(output.bit_dims[0], output.bit_dims[1]);
  output_label_matrix.load(output_label.host_data, output.bit_dims[0],
                           output.bit_dims[1]);

  Host_Matrix not_out = output;

  Launch::kernel_2d(output.element_dims[0], output.element_dims[1]);

  NOT<<<Launch::num_blocks, Launch::threads_per_block>>>(not_out_label);
  SYNC_KERNEL("NOT_out_label");

  NOT<<<Launch::num_blocks, Launch::threads_per_block>>>(not_out);
  SYNC_KERNEL("NOT_output");

  Host_Matrix fp_error = output;
  Host_Matrix fn_error = not_out;

  // fp_error replaces output (which it was set to right above) in the call in
  // order to conform to the LHS/RHS design
  AND<<<Launch::num_blocks, Launch::threads_per_block>>>(fp_error,
                                                         not_out_label);
  SYNC_KERNEL("find_fp_error");

  // see previous comment, but with fn_error and not_out
  AND<<<Launch::num_blocks, Launch::threads_per_block>>>(fn_error,
                                                         output_label_matrix);
  SYNC_KERNEL("find_fn_error");

// fp_error_kernel
// fn_error_kernel
// I think this wants to be __CUDACC__, since __CUDA_ARCH__ is only defined on
// the device and this stuff is on host
#if __CUDACC__ >= 800

  Host_Matrix input_T = input;
  Host_Matrix fp_error_T = fp_error;
  Host_Matrix fn_error_T = fn_error;

  Launch::kernel_2d(input_T.dims[0], input_T.dims[1]);
  Transpose<<<Launch::num_blocks, Launch::threads_per_block>>>(input_T);
  SYNC_KERNEL("Transpose_input");

  Launch::kernel_2d(fp_error_T.dims[0], fp_error_T.dims[1]);
  Transpose<<<Launch::num_blocks, Launch::threads_per_block>>>(fp_error_T);
  SYNC_KERNEL("Transpose_fp_error");

  Launch::kernel_2d(fn_error_T.dims[0], fn_error_T.dims[1]);
  Transpose<<<Launch::num_blocks, Launch::threads_per_block>>>(fn_error_T);
  SYNC_KERNEL("Transpose_fn_error");

  // NOT input transposed
  Host_Matrix not_input_T = input_T;

  Launch::kernel_2d(not_input_T.dims[0], not_input_T.dims[1]);
  NOT<<<Launch::num_blocks, Launch::threads_per_block>>>(not_input_T);
  SYNC_KERNEL("NOT_input_T");

  Launch::calculate_occupancy(FcLayerBkWeight);
  FcLayerBkWeight<<<Launch::num_blocks, Launch::threads_per_block>>>(
      weights, weight_counters, fp_error_t, fn_error_t, input_t, not_input_t);
  SYNC_KERNEL("FcLayerBkWeight_CC80");

#else

  Launch::calculate_occupancy(FcLayerBkWeight);
  FcLayerBkWeight<<<Launch::num_blocks, Launch::threads_per_block>>>(
      input, weights, weight_counters, fp_error, fn_error);
  SYNC_KERNEL("FcLayerBkWeight");

#endif
}

#pragma endregion

#pragma region CvLayer

// void CvLayer::CvLayer(int _input_size, int _output_size, int _batch_size):
// Layer(_input_size, _output_size, _batch_size){

//     return;

// }

// void CvLayer::forward(){
//     //Launch::calculate_occupancy(CvLayerFwd);
//     //FcLayerFwd<<<Launch::num_blocks, Launch::threads_per_block>>>(this);
//     //SYNC_KERNEL(CvLayerFwd);
// }

// void CvLayer::back(){
//     //Launch::calculate_occupancy(CvLayerBk);
//     //FcLayerBk<<<Launch::num_blocks, Launch::threads_per_block>>>(this);
//     //SYNC_KERNEL(CvLayerBk);
// }

#pragma endregion