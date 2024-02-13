#include <iostream>
#include <fstream>
#include <cmath>
#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>	

// Funzione per calcolare il tempo di esecuzione di un kernel CUDA
float elapsedTime(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

void loadKernels(const std::string& filename, float* kernels, int kernel_dims, int out_channels ) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for(int k = 0; k < out_channels; k++) {
        for (int i = 0; i < kernel_dims * kernel_dims * kernel_dims; ++i) {
            if (!(file >> kernels[i + k*kernel_dims*kernel_dims*kernel_dims])) {
                std::cerr << "Error reading file " << filename << std::endl;
                return;
            }
        }
    }
}


void loadBias(const std::string& filename, float* bias,int output_channels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < output_channels; ++i) {
        if (!(file >> bias[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadMeans(const std::string& filename, float* means,int output_channels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < output_channels; ++i) {
        if (!(file >> means[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadVariances(const std::string& filename, float* variances, int output_channels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < output_channels; ++i) {
        if (!(file >> variances[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadWeights(const std::string& filename, float* weights, int output_channels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < output_channels; ++i) {
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

void storeFeatureMaps(const std::string& fname, float *output, int outputRow, int outputCol, int out_channels){
    // Store output    
    std::ofstream outputFile(fname);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return;
    }

    for (int c = 0; c< out_channels; c++){
        outputFile << "Feature map (cuda) #" << c + 1 << std::endl;
        for (int i = 0; i < outputRow; ++i) {
            for (int j = 0; j < outputCol; ++j) {
                outputFile << output[c*outputCol*outputRow + i * outputCol + j] << " ";
            }
            outputFile << std::endl;
        }
        outputFile << std::endl; 
    }

}

void printLoadedKernels(const std::string& filename, float* kernels, int kernel_dims, int out_channels) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return;
    }

    for (int k = 0; k < out_channels; ++k) {
        outputFile << "Kernel #" << k + 1 << std::endl;
        for (int i = 0; i < kernel_dims * kernel_dims * kernel_dims; ++i) {
            outputFile << kernels[i + k * kernel_dims * kernel_dims * kernel_dims] << " ";
        }
        outputFile << std::endl;
    }

    outputFile.close();
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
    int imgRow, imgCol, imgChannels,kernel_dims, padding,outputRow,outputCol,output_channels,stride;
    kernel_dims = 3;
    output_channels = 16;
    padding = 1;
    imgRow = 128 + padding*2;
    imgCol = 192 + padding*2;
    imgChannels = 3;
    stride = 2;
    
    // computing outputRow and outputCol
    outputRow = 1*((imgRow - kernel_dims)/stride+1);
    outputCol = 1*((imgCol - kernel_dims)/stride +1);

    // Allocazione delle variabili host con malloc
    float *image = (float*)malloc(sizeof(float) * imgChannels * imgRow * imgCol);
    float *kernel = (float*)malloc(sizeof(float) * output_channels * kernel_dims * kernel_dims * kernel_dims);
    float *output = (float*)malloc(sizeof(float) * output_channels * outputRow * outputCol);
    float *bias = new float[output_channels];
    float *means = new float[output_channels];
    float *variances = new float[output_channels];
    float *weights = new float[output_channels];

    // Verifica se l'allocazione di memoria è riuscita
    if (image == NULL || kernel == NULL || output == NULL) {
        std::cerr << "Errore nell'allocazione di memoria per le variabili host" << std::endl;
        // Gestire l'errore come preferisci, ad esempio terminando il programma
        exit(EXIT_FAILURE);
    }

    // data load (correct)
    loadImageWithPadding("preprocessed_image.txt", image, imgRow, imgCol, imgChannels); //padding added.
    loadKernels("0.weight.txt", kernel, kernel_dims, output_channels);
    loadBias("1.bias.txt", bias, output_channels);
    loadMeans("1.running_mean.txt", means, output_channels);
    loadVariances("1.running_var.txt", variances, output_channels);
    loadWeights("1.weight.txt",weights,output_channels);

    // // padding check (correct)
    // printImageWithPadding("check_padding.txt", image, imgRow, imgCol, imgChannels);
    // printLoadedKernels("prova_caricamento_kernel.txt", kernel, kernel_dims, output_channels);

    cudaEvent_t start, stop, start_conv,stop_conv,start_shift,stop_shift;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_conv);
    cudaEventCreate(&stop_conv);
    cudaEventCreate(&start_shift);
    cudaEventCreate(&stop_shift);

    // allocating cuda variables
    cudaEventRecord(start);
    cudaEventRecord(start_shift);


    float *d_image, *d_output, *d_kernel, *d_bias, *d_means, *d_variances, *d_weights;


    cudaMalloc((void**)&d_image, sizeof(float) * imgChannels * imgRow * imgCol);
    cudaMalloc((void**)&d_kernel, sizeof(float) * output_channels * kernel_dims * kernel_dims * kernel_dims);
    cudaMalloc((void**)&d_output, sizeof(float) * output_channels * outputRow * outputCol); 
    cudaMalloc((void**)&d_bias, sizeof(float) * output_channels);
    cudaMalloc((void**)&d_means, sizeof(float) * output_channels);
    cudaMalloc((void**)&d_variances, sizeof(float) * output_channels);
    cudaMalloc((void**)&d_weights, sizeof(float) * output_channels);

    cudaMemcpy(d_image, image, sizeof(float) * imgChannels * imgRow * imgCol, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(float) * output_channels * kernel_dims * kernel_dims * kernel_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * output_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, means, sizeof(float) * output_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_variances, variances, sizeof(float) * output_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, sizeof(float) * output_channels, cudaMemcpyHostToDevice);

    cudaEventRecord(stop_shift);

	//vscc
	int threadsPerBlock = 32;

	int gridCols = ceil(float(outputCol) / float(threadsPerBlock));
	int gridRows = ceil(float(outputRow) / float(threadsPerBlock));
    int gridChannels = output_channels;

	dim3 gridDim(gridCols, gridRows,gridChannels);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);

    // Avvia il timer
    cudaEventRecord(start_conv);

	gpuMatrixConv3D << < gridDim, blockDim >> > (d_image, d_kernel, d_weights, d_output, imgRow, imgCol, imgChannels, kernel_dims, outputRow, outputCol,d_bias,d_means,d_variances,stride,stride);

    // CHECK IF IT IS NEEDED.
    cudaDeviceSynchronize();

    // Ferma il timer
    cudaEventSynchronize(stop_conv);
    cudaEventRecord(stop_conv);
    cudaEventRecord(stop);

    // Calcola e stampa il tempo di esecuzione
    float overall = elapsedTime(start, stop);
    float conv = elapsedTime(start_conv, stop_conv);
    float shift = elapsedTime(start_shift, stop_shift);
    std::cout << " Shift time: " << shift << " ms" << std::endl;
    std::cout << " Conv time: " << conv << " ms" << std::endl;
    std::cout << " Overall time: " << overall << " ms" << std::endl;

    // Altri codici rimanenti...
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy the result back to host
    cudaMemcpy(output, d_output, sizeof(float) * output_channels * outputRow * outputCol, cudaMemcpyDeviceToHost);
    // add synchronization if needed.
    // store feature map
    storeFeatureMaps("convlution.txt", output, outputRow, outputCol, output_channels);

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel);
    return 0;
}