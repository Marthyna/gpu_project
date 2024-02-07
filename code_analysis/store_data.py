import torch
import torchvision.transforms as transforms
from Network import Network
from PIL import Image
import os

# Definisci la funzione per il preprocessing dell'immagine
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(128, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image

# Definisci il percorso dell'immagine e del modello
current_directory = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_directory, "imgtest.jpg")
model_path = os.path.join(current_directory, "BEST_MODEL_TRAINED.tar")

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
preprocessed_image = preprocess_image(image_path)

# Salva l'immagine preprocessata
torch.save(preprocessed_image, "preprocessed_image.pth")

# Salva lo stato della stem_conv
torch.save(model.stem_conv.state_dict(), "stem_conv_state_dict.pth")
