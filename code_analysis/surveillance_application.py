import cv2
from PIL import Image
from Network import Network
import torch
import os
import pygame
pygame.mixer.init()

# Funzione per riprodurre il suono di allerta
def play_alert_sound():
    sound_file = os.path.join(current_directory, "alert_sound.wav")
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()


model_architecture = [
    [['InvBottleNeck', 3], ['InvBottleNeck', 3], [0,0,1]],
    [['InvBottleNeck', 3], ['InvBottleNeck', 3], [0,1,0]],
    [['InvBottleNeck', 7], ['ConvNext', 7], [0,1,1]],
    [['ConvNext', 5], ['ConvNext', 5], [1,1,1]]
]

print("Starting camera...")
# Ottieni il percorso del modello
current_directory = os.path.dirname(os.path.abspath(__file__))
trained_model_path = os.path.join(current_directory, "BEST_MODEL_TRAINED.tar")

# Carica il modello precedentemente addestrato
model = Network(model_architecture)
model_state_dict = torch.load(trained_model_path, map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict['model_state_dict'])

# Funzione per la predizione dell'immagine utilizzando il modello
def predict_image(image):
    prediction = model.predict(image)
    return prediction

# Inizializza la telecamera
camera = cv2.VideoCapture(0)

print("PRESS q to quit.")
# Ciclo per acquisire e analizzare le immagini
while True:
    # Leggi l'immagine dalla telecamera
    _, frame = camera.read()

    # Converti l'immagine in formato PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Effettua la predizione sull'immagine
    prediction = predict_image(image)

    # Emetti un suono di allerta e mostra la scritta "ALERT!" rossa se necessario
    if prediction == 'PERSON':
        play_alert_sound()
        cv2.putText(frame, "ALERT!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostra l'immagine acquisita
    cv2.imshow("Camera", frame)

    # Esci dal ciclo se viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la cattura della telecamera e chiudi le finestre
camera.release()
cv2.destroyAllWindows()