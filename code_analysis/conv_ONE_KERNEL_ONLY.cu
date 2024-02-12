#include <iostream>
#include <fstream>
#include <cmath>

void loadKernel(const std::string& filename, float* kernels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < 3 * 3 * 3; ++i) {
        if (!(file >> kernels[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadBias(const std::string& filename, float* bias) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < 16; ++i) {
        if (!(file >> bias[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadMeans(const std::string& filename, float* means) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < 16; ++i) {
        if (!(file >> means[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}

void loadVariances(const std::string& filename, float* variances) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < 16; ++i) {
        if (!(file >> variances[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}


// Funzione per il caricamento delle immagini da un file di testo
void loadImage(const std::string fname, float * img ) {
    std::ifstream imageFile(fname);
    if (!imageFile.is_open()) {
        std::cerr << "Error opening image file " << fname << std::endl;
        return;
    }
    for (int channel = 0; channel < 3; ++channel) {
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 192; ++j) {
                if (!(imageFile >> img[channel * 128 * 192 + i * 192 + j])) {
                    std::cerr << "Error reading image file " << fname << std::endl;
                    return;
                }
            }
        }
    }
}

__global__ void gpuMatrixConv3D(float* image, float* mask, float* result)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0;

	if (row < 64 && col < 96)
	{
		int imageRowsCols = 128 * 192;

		for (int maskRow = 0; maskRow < 3; maskRow++) {
			for (int maskCol = 0; maskCol < 3; maskCol ++) {
				for (int dep = 0; dep < 3; dep++)
            	sum += image[(row + maskRow) * 192 + col + maskCol + dep * imageRowsCols] * mask[maskRow * 3 + maskCol + dep * 3];
			}
		}
		result[row * 96 + col] = sum;
	}
}


int main() {
    int kernel_dims;
    kernel_dims = 3;
    float image[3][128][192];
    float kernel[kernel_dims][kernel_dims][kernel_dims];
    float bias[16];
    float means[16];
    float variances[16];

    loadImage("preprocessed_image.txt", &image[0][0][0]);
    loadKernel("0.weight.txt", &kernel[0][0][0]);
    loadBias("1.weight.txt", bias);
    loadMeans("1.running_mean.txt", means);
    loadVariances("1.running_var.txt", variances);

    float *d_image, *d_output, *d_kernel, *d_bias, *d_means, *d_variances;

    cudaMalloc((void**)&d_image, sizeof(float) * 3 * 128 * 192);
    cudaMalloc((void**)&d_output, sizeof(float) * 16 * 64 * 96); // Output di dimensioni 16x64x96
    cudaMalloc((void**)&d_kernel, sizeof(float) * kernel_dims * kernel_dims * kernel_dims);
    cudaMalloc((void**)&d_bias, sizeof(float) * 16);
    cudaMalloc((void**)&d_means, sizeof(float) * 16);
    cudaMalloc((void**)&d_variances, sizeof(float) * 16);

    cudaMemcpy(d_image, image, sizeof(float) * 3 * 128 * 192, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(float) * kernel_dims * kernel_dims * kernel_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, means, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_variances, variances, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output, sizeof(float) * 64 * 96); // Output di dimensioni 16x64x96

    int threadsPerBlock = 32;

	int gridCols = ceil(float(96) / float(threadsPerBlock));
	int gridRows = ceil(float(64) / float(threadsPerBlock));

	dim3 gridDim(gridCols, gridRows);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);		// total 32*32 = 1024 threads

    gpuMatrixConv3D << < gridDim, blockDim >> > (d_image, d_kernel, d_output);

    std::ofstream outputFile("output.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }

    float output[64][96];
    cudaMemcpy(output, d_output, sizeof(float) * 64 * 96, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 96; ++j) {
            outputFile << output[i][j] << " ";
        }
        outputFile << std::endl;
    }

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_bias);
    cudaFree(d_means);
    cudaFree(d_variances);

    return 0;
}