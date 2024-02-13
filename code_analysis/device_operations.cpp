#include "device_operations.h"

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

void loadBatchParams(const std::string& filename, float* params,int output_channels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < output_channels; ++i) {
        if (!(file >> params[i])) {
            std::cerr << "Error reading file " << filename << std::endl;
            return;
        }
    }
}


void storeConvolution(const std::string& fname, float *output, int outputRow, int outputCol, int out_channels){
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

// image and padding toghether: it works.
void loadImageWithPadding(const std::string& fname, float* img, int imgRows, int imgCol, int imgChannels) {
    std::ifstream imageFile(fname);
    if (!imageFile.is_open()) {
        std::cerr << "Error opening image file " << fname << std::endl;
        return;
    }

    // 0 initialization
    for (int i = 0; i < imgChannels * imgRows * imgCol; ++i) {
        img[i] = 0.0f;
    }

    // Carica i valori dell'immagine originale all'interno del padding
    for (int channel = 0; channel < imgChannels; ++channel) {
        for (int i = 1; i < imgRows - 1; ++i) {
            for (int j = 1; j < imgCol - 1; ++j) {
                if (!(imageFile >> img[channel * imgCol* imgRows + i * imgCol + j])) {
                    std::cerr << "Error reading image file " << fname << std::endl;
                    return;
                }
            }
        }
    }
}

float* loadImage(const std::string& fname, int* imgRows, int* imgCols, int imgChannels) {
    // File opening
    std::ifstream imageFile(fname);
    if (!imageFile.is_open()) {
        std::cerr << "Error opening image file " << fname << std::endl;
        return nullptr;
    }

    // Reading nRows, nCols
    if (!(imageFile >> *imgRows >> *imgCols)) {
        std::cerr << "Error in reading dimensions " << fname << std::endl;
        return nullptr;
    }

    int ImgRows = *imgRows;
    int ImgCols = *imgCols;

    // Allocating space
    float* img = (float*)malloc(sizeof(float) * imgChannels * ImgRows * ImgCols);

    // Load image values
    for (int channel = 0; channel < imgChannels; ++channel) {
        for (int i = 0; i < ImgRows; i++) {
            for (int j = 0; j < ImgCols; j++) {
                // If something goes wrong, just quit.
                if (!(imageFile >> img[channel * ImgCols * ImgRows + i * ImgCols + j])) {
                    std::cerr << "Error reading image file " << fname << std::endl;
                    free(img); // Free the allocated memory before returning nullptr
                    return nullptr;
                }
            }
        }
    }

    return img;
}


float* imgPadding(float* img, int imgRows, int imgCols, int imgChannels, int padding) {
    // Calcolare le nuove dimensioni
    int paddedImgRows = imgRows + 2 * padding;
    int paddedImgCols = imgCols + 2 * padding;

    // Dichiarare nuovo spazio per ospitare l'immagine con il padding
    float* imgPadded = (float*)malloc(sizeof(float) * imgChannels * paddedImgRows * paddedImgCols);

    // Inizializzazione a tutti zeri
    for (int i = 0; i < imgChannels * paddedImgRows * paddedImgCols; ++i) {
        imgPadded[i] = 0.0f;
    }

    // Inserimento dell'immagine all'interno del padding
    for (int channel = 0; channel < imgChannels; channel++) {
        for (int r = 0; r < imgRows; r++) {
            for (int c = 0; c < imgCols; c++) {
                imgPadded[(c + padding) + (r + padding) * paddedImgCols + channel * paddedImgCols * paddedImgRows] = img[c + r * imgCols + channel * imgCols * imgRows];
            }
        }
    }

    // Restituire l'immagine con il padding
    return imgPadded;
}



//--- DEBUG FUNCTIONS ---

void printImageWithPadding(const std::string& fname, float* img, int imgRows, int imgCol, int imgChannels) {
    std::ofstream outFile(fname);
    if (!outFile.is_open()) {
        std::cerr << "Error opening output file " << fname << std::endl;
        return;
    }

    for (int channel = 0; channel < imgChannels; ++channel) {
        for (int i = 0; i < imgRows; ++i) {
            for (int j = 0; j < imgCol; ++j) {
                outFile << img[channel * imgCol * imgRows + i * imgCol + j] << " ";
            }
            outFile << std::endl;
        }
        outFile << std::endl; // Separate channels with an empty line
    }

    outFile.close();
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


