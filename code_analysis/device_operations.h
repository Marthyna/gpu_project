#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <assert.h>
#include <stdlib.h>

void loadKernels(const std::string& filename, float* kernels, int kernel_dims, int out_channels);
void loadBatchParams(const std::string& filename, float* bias, int output_channels);
void loadImageWithPadding(const std::string& fname, float* img, int imgRow, int imgCol, int imgChannels);
void storeConvolution(const std::string& fname, float *output, int outputRow, int outputCol, int out_channels);
float* loadImage(const std::string& fname, int* imgRows, int * imgCol, int imgChannels);
float* imgPadding(float* img, int imgRows, int imgCol,int imgChannels, int padding);

// debug functions
void printLoadedKernels(const std::string& filename, float* kernels, int kernel_dims, int out_channels);
void printImageWithPadding(const std::string& fname, float* img, int imgRow, int imgCol, int imgChannels);
