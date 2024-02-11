#include <iostream>
#include <fstream>

int main() {
    // Apri il file contenente l'immagine preprocessata
    std::ifstream file("preprocessed_image.txt");
    if (!file.is_open()) {
        std::cerr << "Errore nell'apertura del file" << std::endl;
        return 1;
    }

    // Crea un array per l'immagine (3x128x192)
    float image[3][128][192];

    // Leggi e salva i valori dall'array
    for (int channel = 0; channel < 3; ++channel) {
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 192; ++j) {
                file >> image[channel][i][j];
            }
        }
    }

    // Stampa i primi 5 valori di ogni canale per verifica
    std::cout << "Primi 5 valori per canale:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << image[0][i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Chiudi il file
    file.close();

    return 0;
}