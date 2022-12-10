#pragma once

// this will be the file where we put all the includes relating to cuda
// whenever we want to use cuda in a file, we will call this one.
// because it's protected by #pragma once, dont worry about include
// shenanigans, just include it when you need to use cuda

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <mma.h>
