#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void test_attn_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}