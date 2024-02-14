#include "device_operations.h"
#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>	
#include <chrono>


// Funzione per calcolare il tempo di esecuzione di un kernel CUDA
float elapsedTime(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

__global__ void gpuMatrixConv3D(float* image, float* mask, float* weight, float* result, int imageRows, int imageCols, int maskRC, int maskDepth, int resultRows, int resultCols, float* bias, float* mean, float* variance, int strideRows, int strideCols) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.z;

    if (row < resultRows && col < resultCols) {
        int imageRowsCols = imageRows * imageCols;

        float sum = 0.0;

        // Convolution operation
        for (int maskRow = 0; maskRow < maskRC; maskRow++) {
            for (int maskCol = 0; maskCol < maskRC; maskCol++) {
                for (int dep = 0; dep < maskDepth; dep++) {
                    sum += image[(row * strideRows + maskRow) * imageCols + col * strideCols + maskCol + dep * imageRowsCols] * mask[maskRow * maskRC + maskCol + dep * maskRC*maskRC + channel*maskRC*maskRC*maskDepth];
                }   
            }
        }

        // Batch normalization
        float normalized_sum = ((sum - mean[channel]) / sqrtf(variance[channel]))*weight[channel] + bias[channel];

        // ReLU6 activation
        float relu6_output = fminf(fmaxf(normalized_sum, 0.0f), 6.0f);

        // Store the result
        result[channel*resultCols*resultRows + row * resultCols + col] = relu6_output;
    }
}


int main() {
    // Dimension declaration and definition
    int imgRow, imgCol, imgChannels,kernel_dims,padding,output_channels,stride;
    kernel_dims = 3;
    output_channels = 16;
    padding = 1;
    imgChannels = 3;
    stride = 2;
    
    // model allocation in HOST (cpu)
    float *kernel = (float*) malloc(sizeof(float) * output_channels * kernel_dims * kernel_dims * kernel_dims);
    float *bias = new float[output_channels];
    float *means = new float[output_channels];
    float *variances = new float[output_channels];
    float *weights = new float[output_channels];

    // model loading in HOST (cpu). Same whatever the image is.
    loadKernels("./model/stem_params/0.weight.txt", kernel, kernel_dims, output_channels);
    loadBatchParams("./model/stem_params/1.weight.txt",weights,output_channels);
    loadBatchParams("./model/stem_params/1.bias.txt", bias, output_channels);
    loadBatchParams("./model/stem_params/1.running_mean.txt", means, output_channels);
    loadBatchParams("./model/stem_params/1.running_var.txt", variances, output_channels);


    // loading image...
    float* image = loadImage("./images_processed/imgtest.txt" ,&imgRow, &imgCol, imgChannels);
    
    // computing outputRow and outputCol and then space allocation to store feature maps.
    int outputRow = (imgRow + 2*padding - kernel_dims)/stride +1;
    int outputCol = (imgCol +2*padding - kernel_dims)/stride +1;
    float *output = (float*)malloc(sizeof(float) * output_channels * outputRow * outputCol);

    // padding
    auto startPadding = std::chrono::high_resolution_clock::now();
    image = imgPadding(image, imgRow, imgCol, imgChannels, padding);
    auto endPadding = std::chrono::high_resolution_clock::now();
    
    // Calcola la durata dell'operazione di padding
    std::chrono::duration<float> durationPadding = endPadding - startPadding;
    float paddingDuration_ms = durationPadding.count()*1000;

    // Final allocation check
    if (image == NULL || kernel == NULL || output == NULL) {
        std::cerr << "Allocation (host) error." << std::endl;
        exit(EXIT_FAILURE);
    }

    // declaring device pointers.
    float *d_image, *d_output, *d_kernel, *d_bias, *d_means, *d_variances, *d_weights;

    // allocating space on the device
    cudaMalloc((void**)&d_image, sizeof(float) * imgChannels * (imgRow + 2*padding) * (imgCol + 2*padding));
    cudaMalloc((void**)&d_kernel, sizeof(float) * output_channels * kernel_dims * kernel_dims * kernel_dims);
    cudaMalloc((void**)&d_output, sizeof(float) * output_channels * outputRow * outputCol); 
    cudaMalloc((void**)&d_bias, sizeof(float) * output_channels);
    cudaMalloc((void**)&d_means, sizeof(float) * output_channels);
    cudaMalloc((void**)&d_variances, sizeof(float) * output_channels);
    cudaMalloc((void**)&d_weights, sizeof(float) * output_channels);
    
    // shift
    cudaMemcpy(d_image, image, sizeof(float) * imgChannels * (imgRow + 2*padding) * (imgCol + 2*padding), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(float) * output_channels * kernel_dims * kernel_dims * kernel_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * output_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, means, sizeof(float) * output_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_variances, variances, sizeof(float) * output_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, sizeof(float) * output_channels, cudaMemcpyHostToDevice);

	//grid setup.
	int threadsPerBlock = 16;
	int gridCols = ceil(float(outputCol) / float(threadsPerBlock));
	int gridRows = ceil(float(outputRow) / float(threadsPerBlock));
    int gridChannels = output_channels;
	dim3 gridDim(gridCols, gridRows,gridChannels);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);

    // starting convolution (paralel,gpu)
	gpuMatrixConv3D << < gridDim, blockDim >> > (d_image, d_kernel, d_weights, d_output, imgRow + 2*padding, imgCol + 2*padding, imgChannels, kernel_dims, outputRow, outputCol,d_bias,d_means,d_variances,stride,stride);

    // waiting cuda get the job done to store.
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(output, d_output, sizeof(float) * output_channels * outputRow * outputCol, cudaMemcpyDeviceToHost);
    // store feature map
    storeConvolution("./test_output/convolution_results/stem_conv.txt", output, outputRow, outputCol, output_channels);

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_means);
    cudaFree(d_variances);
    return 0;
}