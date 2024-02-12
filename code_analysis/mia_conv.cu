#include <iostream>
#include <fstream>
#include <cmath>

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

void storeFeatureMap2(const std::string& fname, float *output, int outputRow, int outputCol){
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

__global__ void gpuMatrixConv3DAtomic(float* image, float* kernel, float* output, int kernel_dim,  int imgRow,int imgCol, int stride){

    int kernel_elm,img_elm,mult;
    // accesso al kernel (da scrivere ancora in forma cuda)
    kernel_elm = kernel[threadIdx.z * kernel_dim * kernel_dim + threadIdx.y * kernel_dim + threadIdx.x];

    //accesso all'immagine 
    img_elm = image[ threadIdx.z  * imgRow * imgCol + (stride* blockIdx.y + threadIdx.y) * imgRow +  stride * blockIdx.x + threadIdx.x];

    // moltiplicazione elemento per elemento
    mult = img_elm * kernel_elm;

    // RACE CONDITION! Per scrivere sull'output, serve un operazione ATOMICA!
    atomicAdd(&output[blockIdx.y * blockDim.x + blockIdx.x], mult);
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
    int imgRow, imgCol, imgChannels,kernel_channels,kernel_dims, padding,outputRow,outputCol;
    kernel_dims = 3;
    kernel_channels = 16;
    padding = 1;
    imgRow = 128 + padding*2;
    imgCol = 192 + padding*2;
    imgChannels = 3;
    outputRow = 64;
    outputCol = 96;

    // allocating host variables
    float *image = new float[imgChannels * imgRow * imgCol];
    float *kernel = new float[kernel_dims * kernel_dims * kernel_dims];
    float *bias = new float[kernel_channels];
    float *means = new float[kernel_channels];
    float *variances = new float[kernel_channels];
    float *output = new float[outputRow * outputCol];

    // loading values (created through pytorch). Outside the purpouse of the parallelization.
    loadImageWithPadding("preprocessed_image.txt", image, imgRow, imgCol, imgChannels); //padding added.
    loadKernel("0.weight.txt", kernel, kernel_dims);
    loadBias("1.weight.txt", bias, kernel_channels);
    loadMeans("1.running_mean.txt", means, kernel_channels);
    loadVariances("1.running_var.txt", variances, kernel_channels);

    // padding check
    printImageWithPadding("check_padding.txt", image, imgRow, imgCol, imgChannels);

    // allocating cuda variables
    float *d_image, *d_output, *d_kernel, *d_bias, *d_means, *d_variances;

    cudaMalloc((void**)&d_image, sizeof(float) * imgChannels * imgRow * imgCol);
    cudaMalloc((void**)&d_output, sizeof(float) * kernel_channels* outputRow * outputCol); 
    cudaMalloc((void**)&d_kernel, sizeof(float) * kernel_dims * kernel_dims * kernel_dims);
    cudaMalloc((void**)&d_bias, sizeof(float) * kernel_channels);
    cudaMalloc((void**)&d_means, sizeof(float) * kernel_channels);
    cudaMalloc((void**)&d_variances, sizeof(float) * kernel_channels);

    cudaMemcpy(d_image, image, sizeof(float) * imgChannels * imgRow * imgCol, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(float) * kernel_dims * kernel_dims * kernel_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * kernel_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, means, sizeof(float) * kernel_channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_variances, variances, sizeof(float) * kernel_channels, cudaMemcpyHostToDevice);

    
    dim3 gridDim(outputCol, outputRow);
    dim3 blockDim(3, 3, 3); 
    
    // kernel call
    gpuMatrixConv3DAtomic<<<gridDim, blockDim>>>(d_image, d_kernel, d_output, kernel_dims, imgRow,imgCol, 2);

    // store output on host (CPU)
    cudaMemcpy(output, d_output, sizeof(float) * outputRow * outputCol, cudaMemcpyDeviceToHost);

    // store feature map
    storeFeatureMap2("nonpuoentrare.txt", output, outputRow, outputCol);

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_bias);
    cudaFree(d_means);
    cudaFree(d_variances);

    delete[] image;
    delete[] kernel;
    delete[] bias;
    delete[] means;
    delete[] variances;
    delete[] output;

    return 0;
}
