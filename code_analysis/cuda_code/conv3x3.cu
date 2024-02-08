#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
using namespace std::chrono;

// change these values according to image size
#define N 0
#define WIDTH 0
#define HEIGHT 0

__global__ void conv3x3_kernel(float *input, float *output, float *kernel, int input_width, int input_height, int output_width, int output_height) {    // Get global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = idy * output_width + idx;

    if (idx < width && idy < height) {

        // CONVD2
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int in_idx = (idy * stride + ky) * input_width + (idx * stride + kx);
                int kernel_idx = ky * kernel_width + kx;
                sum += input[in_idx] * kernel[kernel_idx];
            }
        }
        output[out_idx] = sum;

        // BATCHNORM
        output[idx] = scale[idx] * (input[idx] - mean[idx]) / sqrtf(variance[idx] + epsilon) + shift[idx];
    
        // RELU
        output[idx] = fmaxf(output[idx], 0.0f);
    }
}

void conv3x3(float *input, float *output, int width, int height) {
    // Define block and grid dimensions
    int maxBlockDimx = deviceProp.maxThreadsDim[0];
    int maxBlockDimy = deviceProp.maxThreadsDim[1];

    int block_x = std::min(maxBlockDimx, N);
    int block_y = std::min(maxBlockDimy, N);
    dim3 blockDim(block_x, block_y);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel
    conv3x3_kernel<<<gridDim, blockDim>>>(input, output, width, height);

    cudaEventRecord(stop, 0);

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();
}

int main() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Declare and init CUDA event vars
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on host and device
    float *h_input, *h_output;
    float *d_input, *d_output;
    int width = WIDTH, height = HEIGHT;

    // Allocate memory on host
    h_input = new float[width * height];
    h_output = new float[width * height];

    // Allocate memory on device
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    // Transfer data from host to device
    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution operation
    conv3x3(d_input, d_output, width, height);

    // Transfer data from device to host
    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Make sure event has finished and get its value
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    // Get time between two events
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Check errors at runtime
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    // Free memory
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
