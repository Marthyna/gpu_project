import torch

# Carica il file .pth
preprocessed_image = torch.load("preprocessed_image.pth")

# Stampiamo il tipo e le dimensioni del tensore
print("Tipo del tensore:", type(preprocessed_image))
print("Dimensioni del tensore:", preprocessed_image.size())