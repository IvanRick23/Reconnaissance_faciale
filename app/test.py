import face_recognition
from PIL import Image
from io import BytesIO
import numpy as np

# Chargez une de vos images de test (remplacez 'chemin/vers/votre_image.jpg')
with open('C:\\Users\\Achille\\Desktop\\kengne\\IMG_2555 (1).JPG', 'rb') as f:
    image_bytes = f.read()

# Logique de la fonction get_face_encoding :
image = Image.open(BytesIO(image_bytes))
image_np = np.array(image.convert("RGB"))

face_locations = face_recognition.face_locations(image_np)

print(f"Nombre de visages trouvés : {len(face_locations)}")
if face_locations:
    print("Visage détecté ! Le problème est dans le pipeline de l'API.")
else:
    print("ÉCHEC de la détection. Le problème est l'installation de dlib/face_recognition.")