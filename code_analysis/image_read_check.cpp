#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

int main() {
    // Carica il file della matrice preprocessata
    torch::Tensor preprocessed_image = torch::zeros({1, 3, 128, 128});
    try {
        preprocessed_image = torch::load("preprocessed_image.pth");
    } catch (const c10::Error& e) {
        std::cerr << "Errore durante il caricamento del file: " << e.what() << std::endl;
        return -1;
    }

    // Stampa la matrice preprocessata
    std::cout << "Matrice preprocessata:" << std::endl;
    std::cout << preprocessed_image << std::endl;

    return 0;
}
