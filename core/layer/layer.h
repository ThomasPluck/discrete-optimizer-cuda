#pragma once

#include"standard_includes.h"

#include"../structs/structs.h"
#include"macros.h"
#include"launch.h"
#include"kernel.cuh"

//if you want to use this, replace the arguments for the ints with an argument for an object 
//of this type, then pass in *this to the kernel call in place of the ints

struct Param;

struct Layer {
    Host_Matrix input;
    Host_Matrix output;

    Host_Matrix input_label;
    Host_Matrix output_label;

    // Blocks according to row-major/col-major nature of Matrix
    int input_blocks;
    int output_blocks;
    int batch_blocks;

    // Number of bits required to span bit matrices
    int input_bits;
    int output_bits;
    int batch_bits;

    Layer(int _input_size, int _output_size, int _batch_size);

    virtual void forward(){}

    virtual void back(){}

    operator Param();
};

struct Param {
    int input_blocks, output_blocks, batch_blocks;
    int input_bits, output_bits, batch_bits;

    Param(Layer* input);
};

struct FcLayer : public Layer {

    Host_Matrix weights;

    Host_Data<uint16_t> biases;
    Host_Data<uchar> weight_counters;
    Host_Data<uchar> bias_counters;

    // input_size, output_size, batch_size are taken 
    FcLayer(int _input_size, int _output_size, int _batch_size);

    void forward();

    void back();


};

struct CvLayer : public Layer {

    //TODO: Correctly implement data loading
    Host_Matrix filters;
    Host_Data<uint16_t> biases;
    Host_Data<uchar> filter_counters;
    Host_Data<uchar> bias_counters;

    // Striding dimensions
    int stride_width;
    int stride_height;
    // Padding dimensions
    int padding_width;
    int padding_height;
    // Filter dimensions
    int filter_width;
    int filter_height;
    // Pooling parameters
    int pooling_width;
    int pooling_height;

    CvLayer(int _input_size, int _output_size, int _batch_size);

    void forward();

    void back();

};

void link(Layer* prelink, Layer* postlink);