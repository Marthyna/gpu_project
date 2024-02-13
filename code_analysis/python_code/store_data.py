import torch
import numpy as np
import os
from PIL import Image
from Network import Network
from torchvision import transforms


# Funzione per il caricamento dell'immagine
def load_image(image_path):
    image = Image.open(image_path)
    return image


# Funzione per il preprocessing dell'immagine
def preprocess(image : Image.Image):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype( dtype = torch.float ),
        transforms.Resize(128,antialias=True),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] )
    ])
    X = transform(image).unsqueeze(0)
    return X

# Definisci il percorso dell'immagine e del modello
current_directory = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_directory, "../images/imgtest.jpg")
model_path = os.path.join(current_directory, "../model/BEST_MODEL_TRAINED.tar")

# Carica il modello
model_architecture = [
    [['InvBottleNeck', 3], ['InvBottleNeck', 3], [0,0,1]],
    [['InvBottleNeck', 3], ['InvBottleNeck', 3], [0,1,0]],
    [['InvBottleNeck', 7], ['ConvNext', 7], [0,1,1]],
    [['ConvNext', 5], ['ConvNext', 5], [1,1,1]]
]
device = "cpu"
model = Network(model_architecture, device)
model_state_dict = torch.load(model_path, map_location=torch.device(device))
model.load_state_dict(model_state_dict['model_state_dict'])

# Preprocessa l'immagine
preprocessed_image = preprocess(load_image(image_path))

# Salva l'immagine preprocessata in un file di testo in un formato leggibile per C++
with open("../images_processed/imgtest.txt", 'w') as f:
    print(preprocessed_image.shape[2],preprocessed_image.shape[3],file=f)
    for channel in preprocessed_image:
        for row in channel:
            np.savetxt(f, row.numpy(), fmt='%f')
print(preprocessed_image.shape)

# Salva i parametri di stem_convolution in un file di testo in un formato leggibile per C++
stem_conv_params = model.stem_conv.state_dict()
for name, param in stem_conv_params.items():
        # Salva i tensori in formato leggibile per C++
        if name != "1.num_batches_tracked":
            with open("../model/stem_params/" + name + ".txt", 'w') as f:
                if len(param.size()) == 4:
                    for channel in param:
                        for matrix in channel:
                            np.savetxt(f, matrix.numpy(), fmt='%f')
                        f.write('\n')
                else:
                    np.savetxt(f, param.numpy(), fmt='%f')

# print (stem_conv_params)