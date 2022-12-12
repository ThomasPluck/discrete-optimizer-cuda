#pragma once

#include "standard_includes.h"

#include "../structs/structs.h"
#include "kernel.cuh"
#include "launch.h"
#include "macros.h"

#include "layer.h"

struct Sequential {

  // Data attributes
  std::vector<Layer> layers;
  Host_Matrix input;
  Host_Matrix output;
  Host_Matrix output_label;

  // Parameter attributes
  int batch_blocks;
  int batch_bits;

  Sequential(std::vector<Layer> _layers, int _batch_size);

  void predict() {}

  void train() {}
};