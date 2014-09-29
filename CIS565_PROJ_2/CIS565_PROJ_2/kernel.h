#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BLOCK_SIZE 128

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);