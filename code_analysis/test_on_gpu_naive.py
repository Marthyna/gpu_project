from PIL import Image
from Network import Network
import torch
import os
import time
from metrics import get_macs_and_params

# Funzione per il caricamento dell'immagine
def load_image(image_path):
    image = Image.open(image_path)
    return image

# Inizializza il percorso dell'immagine
current_directory = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_directory, "imgtest.jpg")

# Carica l'immagine
image = load_image(image_path)

# Ottieni il percorso del modello
trained_model_path = os.path.join(current_directory, "BEST_MODEL_TRAINED.tar")

# Carica il modello precedentemente addestrato
model_architecture = [
    [['InvBottleNeck', 3], ['InvBottleNeck', 3], [0,0,1]],
    [['InvBottleNeck', 3], ['InvBottleNeck', 3], [0,1,0]],
    [['InvBottleNeck', 7], ['ConvNext', 7], [0,1,1]],
    [['ConvNext', 5], ['ConvNext', 5], [1,1,1]]
]

device = "cuda"

model = Network(model_architecture,device)
model_state_dict = torch.load(trained_model_path, map_location=torch.device(device))
model.load_state_dict(model_state_dict['model_state_dict'])

# Funzione per la predizione dell'immagine utilizzando il modello
def predict_image(image):
    start_time = time.time()
    prediction = model.predict(image)
    end_time = time.time()
    inference_time = end_time - start_time
    return prediction, inference_time

# Effettua la predizione sull'immagine
prediction, inference_time = predict_image(image)
# Stampa il risultato della predizione e il tempo impiegato per l'inferenza
print(f"---------------{model.device}---------------------")
print("Total Inference Time:", inference_time, "seconds")
id,block,t = model.info_bottleneck()
print ( f" critical block = {model_architecture[id]}, f_time = {t}s" )
model.info_time()
print(model.stem_conv.state_dict())