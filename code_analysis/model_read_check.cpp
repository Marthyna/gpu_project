#include <iostream>
#include <fstream>
#include <vector>

// Funzione per il caricamento dei kernel da un file di testo
void loadKernels(const std::string& filename, float kernels[16][3][3]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore durante l'apertura del file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                if (!(file >> kernels[i][j][k])) {
                    std::cerr << "Errore durante la lettura del file " << filename << std::endl;
                    return;
                }
            }
        }
    }
}

// Funzione per il caricamento dei bias da un file di testo
void loadBias(const std::string& filename, float bias[16]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore durante l'apertura del file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < 16; ++i) {
        if (!(file >> bias[i])) {
            std::cerr << "Errore durante la lettura del file " << filename << std::endl;
            return;
        }
    }
}

// Funzione per il caricamento delle medie da un file di testo
void loadMeans(const std::string& filename, float means[16]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore durante l'apertura del file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < 16; ++i) {
        if (!(file >> means[i])) {
            std::cerr << "Errore durante la lettura del file " << filename << std::endl;
            return;
        }
    }
}

// Funzione per il caricamento delle varianze da un file di testo
void loadVariances(const std::string& filename, float variances[16]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore durante l'apertura del file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < 16; ++i) {
        if (!(file >> variances[i])) {
            std::cerr << "Errore durante la lettura del file " << filename << std::endl;
            return;
        }
    }
}

// Funzione per la stampa dei kernel
void printKernels(const float kernels[16][3][3]) {
    for (int i = 0; i < 16; ++i) {
        std::cout << "Kernel " << i << ":" << std::endl;
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                std::cout << kernels[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// Funzione per la stampa dei bias
void printBias(const float bias[16]) {
    std::cout << "Bias:" << std::endl;
    for (int i = 0; i < 16; ++i) {
        std::cout << bias[i] << " ";
    }
    std::cout << std::endl;
}

// Funzione per la stampa delle medie
void printMeans(const float means[16]) {
    std::cout << "Means:" << std::endl;
    for (int i = 0; i < 16; ++i) {
        std::cout << means[i] << " ";
    }
    std::cout << std::endl;
}

// Funzione per la stampa delle varianze
void printVariances(const float variances[16]) {
    std::cout << "Variances:" << std::endl;
    for (int i = 0; i < 16; ++i) {
        std::cout << variances[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    float kernels[16][3][3];
    float bias[16];
    float means[16];
    float variances[16];

    // Carica i kernels
    loadKernels("0.weight.txt", kernels);

    // Carica i bias
    loadBias("1.weight.txt", bias);

    // Carica le medie
    loadMeans("1.running_mean.txt", means);

    // Carica le varianze
    loadVariances("1.running_var.txt", variances);

    // Stampa i kernels
    printKernels(kernels);

    // Stampa i bias
    printBias(bias);

    // Stampa le medie
    printMeans(means);

    // Stampa le varianze
    printVariances(variances);

    return 0;
}