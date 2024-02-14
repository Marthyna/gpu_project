#include <iostream>
#include <fstream>
#include <cmath>
#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>	

void loadKernel(const std::string& filename, float* kernels, int kernel_dims ) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < kernel_dims * kernel_dims * kernel_dims; ++i) {
        if (!(file >> kernels[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadBias(const std::string& filename, float* bias,int kernel_channels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < kernel_channels; ++i) {
        if (!(file >> bias[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadMeans(const std::string& filename, float* means,int kernel_channels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < kernel_channels; ++i) {
        if (!(file >> means[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadVariances(const std::string& filename, float* variances, int kernel_channels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < kernel_channels; ++i) {
        if (!(file >> variances[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadWeights(const std::string& filename, float* weights, int kernel_channels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < kernel_channels; ++i) {
        if (!(file >> weights[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadImageWithPadding(const std::string& fname, float* img, int imgRow, int imgCol, int imgChannels) {
    std::ifstream imageFile(fname);
    if (!imageFile.is_open()) {
        std::cerr << "Error opening image file " << fname << std::endl;
        return;
    }

    // reading of nRows and nCols. In this code we only
    // focus on make the convolution working.
    // thus we skip row and cols and we use 
    // predefined sizes.
    int skipNRow, skipNCol;
    imageFile >> skipNRow >> skipNCol;

    // 0 initialization
    for (int i = 0; i < imgChannels * imgRow * imgCol; ++i) {
        img[i] = 0.0f;
    }

    // Carica i valori dell'immagine originale all'interno del padding
    for (int channel = 0; channel < imgChannels; ++channel) {
        for (int i = 1; i < imgRow - 1; ++i) {
            for (int j = 1; j < imgCol - 1; ++j) {
                if (!(imageFile >> img[channel * imgCol* imgRow + i * imgCol + j])) {
                    std::cerr << "Error reading image file " << fname << std::endl;
                    return;
                }
            }
        }
    }
}

void storeFeatureMap(const std::string& fname, float *output, int outputRow, int outputCol){
       // Store output    
    std::ofstream outputFile(fname);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return;
    }

    for (int i = 0; i < outputRow; ++i) {
        for (int j = 0; j < outputCol; ++j) {
            outputFile << output[i * outputCol + j] << " ";
        }
        outputFile << std::endl;
    }

}

__global__ void gpuMatrixConv3D(float* image, float* mask, float* weight, float* result, int imageRows, int imageCols, int maskRC, int maskDepth, int resultRows, int resultCols, float* bias, float* mean, float* variance, int strideRows, int strideCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < resultRows && col < resultCols) {
        int imageRowsCols = imageRows * imageCols;

        float sum = 0.0;

        // Convolution operation
        for (int maskRow = 0; maskRow < maskRC; maskRow++) {
            for (int maskCol = 0; maskCol < maskRC; maskCol++) {
                for (int dep = 0; dep < maskDepth; dep++) {
                    sum += image[(row * strideRows + maskRow) * imageCols + col * strideCols + maskCol + dep * imageRowsCols] * mask[maskRow * maskRC + maskCol + dep * maskDepth*maskDepth];
                }   
            }
        }

        // Batch normalization
        float normalized_sum = ((sum - mean[0]) / sqrtf(variance[0]))*weight[0] + bias[0];

        // ReLU6 activation
        float relu6_output = fminf(fmaxf(normalized_sum, 0.0f), 6.0f);

        // Store the result
        result[row * resultCols + col] = relu6_output;
    }
}



void printImageWithPadding(const std::string& fname, float* img, int imgRow, int imgCol, int imgChannels) {
    std::ofstream outFile(fname);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file " << fname << std::endl;
        return;
    }

    for (int channel = 0; channel < imgChannels; ++channel) {
        for (int i = 0; i < imgRow; ++i) {
            for (int j = 0; j < imgCol; ++j) {
                outFile << img[channel * imgCol * imgRow + i * imgCol + j] << " ";
            }
            outFile << std::endl;
        }
        outFile << std::endl; // Separate channels with an empty line
    }

    outFile.close();
}


int main() {
    // Dimension declaration and definition
    int imgRow, imgCol, imgChannels,kernel_dims, padding,outputRow,outputCol,kernel_channels;
    kernel_dims = 3;
    kernel_channels = 16;
    padding = 1;
    imgRow = 128 + padding*2;
    imgCol = 192 + padding*2;
    imgChannels = 3;
    outputRow = 64;
    outputCol = 96;

    // Allocazione delle variabili host con malloc
    float *image = (float*)malloc(sizeof(float) * imgChannels * imgRow * imgCol);
    float *kernel = (float*)malloc(sizeof(float) * kernel_dims * kernel_dims * kernel_dims);
    float *output = (float*)malloc(sizeof(float) * outputRow * outputCol);
    float *bias = new float[kernel_channels];
    float *means = new float[kernel_channels];
    float *variances = new float[kernel_channels];
    float *weights = new float[kernel_channels];

    // allocation check
    if (image == NULL || kernel == NULL || output == NULL) {
        std::cerr << "Errore nell'allocazione di memoria per le variabili host" << std::endl;
        exit(EXIT_FAILURE);
    }

    // data load (correct)
    loadImageWithPadding("./images_processed/imgtest.txt", image, imgRow, imgCol, imgChannels); //padding added.
    loadKernel("./model/stem_params/0.weight.txt", kernel, kernel_dims);
    loadBias("./model/stem_params/1.bias.txt", bias, kernel_channels);
    loadMeans("./model/stem_params/1.running_mean.txt", means, kernel_channels);
    loadVariances("./model/stem_params/1.running_var.txt", variances, kernel_channels);
    loadWeights("./model/stem_params/1.weight.txt",weights,kernel_channels);

    // padding check (correct)
    printImageWithPadding("./test_output/data_load_check/check_padding.txt", image, imgRow, imgCol, imgChannels);

    // allocating cuda variables
    float *d_image, *d_output, *d_kernel,*d_bias, *d_means, *d_variances, *d_weights;

    cudaMalloc((void**)&d_image, sizeof(float) * imgChannels * imgRow * imgCol);
    cudaMalloc((void**)&d_kernel, sizeof(float) * kernel_dims * kernel_dims * kernel_dims);
    cudaMalloc((void**)&d_output, sizeof(float) * outputRow * outputCol); 
    cudaMalloc((void**)&d_bias, sizeof(float) * kernel_channels);
    cudaMalloc((void**)&d_means, sizeof(float) * kernel_channels);
    cudaMalloc((void**)&d_variances, sizeof(float) * kernel_channels);
    cudaMalloc((void**)&d_weights, sizeof(float) * kernel_channels);

    cudaMemcpy(d_image, image, sizeof(float) * imgChannels * imgRow * imgCol, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(float) * kernel_dims * kernel_dims * kernel_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * kernel_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, means, sizeof(float) * kernel_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_variances, variances, sizeof(float) * kernel_channels, cudaMemcpyHostToDevice);
     cudaMemcpy(d_weights, weights, sizeof(float) * kernel_channels, cudaMemcpyHostToDevice);

	int threadsPerBlock = 32;

	int gridCols = ceil(float(outputCol) / float(threadsPerBlock));
	int gridRows = ceil(float(outputRow) / float(threadsPerBlock));

	dim3 gridDim(gridCols, gridRows);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);

	gpuMatrixConv3D << < gridDim, blockDim >> > (d_image, d_kernel, &d_weights[0] ,d_output, imgRow, imgCol, imgChannels, kernel_dims, outputRow, outputCol,d_bias,d_means,d_variances,2,2);
    cudaDeviceSynchronize();
    // Copy the result back to host
    cudaMemcpy(output, d_output, sizeof(float) * outputRow * outputCol, cudaMemcpyDeviceToHost);
    // store feature map
    storeFeatureMap("./test_output/convolution_results/feature_map/cuda_our_feature_map.txt", output, outputRow, outputCol);

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_means);
    cudaFree(d_variances);
    return 0;
}
