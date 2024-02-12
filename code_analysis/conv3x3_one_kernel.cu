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


__global__ void gpuMatrixConv3D(float* image, float* mask, float* result,int imgRow, int imgCol, int imgChannels, int kernel_dims, int outputRow,int outputCol)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0;

	if (row < outputRow && col < outputCol)
	{
		int imageRowsCols = imgRow * imgCol;

		for (int maskRow = 0; maskRow < kernel_dims; maskRow++) {
			for (int maskCol = 0; maskCol < kernel_dims; maskCol ++) {
				for (int dep = 0; dep < kernel_dims; dep++)
            	sum += image[(row + maskRow) * imgCol + col + maskCol + dep * imageRowsCols] * mask[maskRow * kernel_dims + maskCol + dep * kernel_dims];
			}
		}
		result[row * outputCol + col] = sum;
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
    int imgRow, imgCol, imgChannels,kernel_channels,kernel_dims, padding,outputRow,outputCol;
    kernel_dims = 3;
    kernel_channels = 16;
    padding = 1;
    imgRow = 128 + padding*2;
    imgCol = 192 + padding*2;
    imgChannels = 3;
    outputRow = 64;
    outputCol = 96;

    float *image = new float[imgChannels * imgRow * imgCol];
    float *kernel = new float[kernel_dims * kernel_dims * kernel_dims];
    float *bias = new float[kernel_channels];
    float *means = new float[kernel_channels];
    float *variances = new float[kernel_channels];
    float *output = new float[outputRow * outputCol];

    loadImageWithPadding("preprocessed_image.txt", image, imgRow, imgCol, imgChannels);
    loadKernel("0.weight.txt", kernel, kernel_dims);
    loadBias("1.weight.txt", bias, kernel_channels);
    loadMeans("1.running_mean.txt", means, kernel_channels);
    loadVariances("1.running_var.txt", variances, kernel_channels);

    // padding check
    printImageWithPadding("check_padding.txt", image, imgRow, imgCol, imgChannels);


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

    int threadsPerBlock = 32;

    int gridCols = ceil(float(outputCol) / float(threadsPerBlock));
    int gridRows = ceil(float(outputRow) / float(threadsPerBlock));

    dim3 gridDim(gridCols, gridRows);
    dim3 blockDim(threadsPerBlock, threadsPerBlock); 

    gpuMatrixConv3D<<<gridDim, blockDim>>>(d_image, d_kernel, d_output, imgRow, imgCol, imgChannels, kernel_dims, outputRow, outputCol);
    cudaMemcpy(output, d_output, sizeof(float) * outputRow * outputCol, cudaMemcpyDeviceToHost);

    // store feature map
    storeFeatureMap("featuremap.txt", output, outputRow, outputCol);

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
