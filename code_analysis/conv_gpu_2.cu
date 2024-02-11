#include <iostream>
#include <fstream>
#include <cmath>

__global__ void conv3x3(float* input, float* output, float* kernels, float* bias,
                         float* means, float* variances, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int y = idx / (128 * 192);
    int x = (idx % (128 * 192)) / 192;
    int channel = idx % 192;

    output[idx] = 0.0;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int input_y = y + i;
            int input_x = x + j;

            input_y = fminf(fmaxf(input_y, 0), 127);
            input_x = fminf(fmaxf(input_x, 0), 191);

            int input_idx = (channel * 128 + input_y) * 192 + input_x;

            output[idx] += kernels[channel * 9 + (i + 1) * 3 + (j + 1)] * input[input_idx];
        }
    }

    output[idx] = (output[idx] - means[channel]) / sqrtf(variances[channel] + epsilon);
    output[idx] += bias[channel];
    output[idx] = fminf(fmaxf(output[idx], 0.0), 6.0);
}

void loadKernels(const std::string& filename, float* kernels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < 16 * 3 * 3; ++i) {
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

int main() {
    std::ifstream imageFile("preprocessed_image.txt");
    float image[3][128][192];
    for (int channel = 0; channel < 3; ++channel) {
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 192; ++j) {
                imageFile >> image[channel][i][j];
            }
        }
    }

    float kernels[16][3][3];
    float bias[16];
    float means[16];
    float variances[16];

    loadKernels("0.weight.txt", kernels[0][0]);
    loadBias("1.weight.txt", bias);
    loadMeans("1.running_mean.txt", means);
    loadVariances("1.running_var.txt", variances);

    float *d_image, *d_output, *d_kernels, *d_bias, *d_means, *d_variances;

    cudaMalloc((void**)&d_image, sizeof(float) * 3 * 128 * 192);
    cudaMalloc((void**)&d_output, sizeof(float) * 16 * 128 * 192);
    cudaMalloc((void**)&d_kernels, sizeof(float) * 16 * 3 * 3);
    cudaMalloc((void**)&d_bias, sizeof(float) * 16);
    cudaMalloc((void**)&d_means, sizeof(float) * 16);
    cudaMalloc((void**)&d_variances, sizeof(float) * 16);

    cudaMemcpy(d_image, image, sizeof(float) * 3 * 128 * 192, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, means, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_variances, variances, sizeof(float) * 16, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 8, 1);
    dim3 gridDim(192 * 128 / 128, 1, 1);

    conv3x3<<<gridDim, blockDim>>>(d_image, d_output, d_kernels, d_bias, d_means, d_variances, 1e-5);

    float output[16][128][192];
    cudaMemcpy(output, d_output, sizeof(float) * 16 * 128 * 192, cudaMemcpyDeviceToHost);

    std::ofstream outputFile("output.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }

    for (int k = 0; k < 16; ++k) {
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 192; ++j) {
                outputFile << output[k][i][j] << " ";
            }
            outputFile << std::endl;
        }
    }

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernels);
    cudaFree(d_bias);
    cudaFree(d_means);
    cudaFree(d_variances);

    return 0;
}
