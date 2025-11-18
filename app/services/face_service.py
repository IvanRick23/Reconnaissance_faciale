
import face_recognition
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Optional
from fastapi import HTTPException, UploadFile

# Importation des fonctions de stockage
from ..db.storage import add_person_to_db, get_all_encodings, delete_person_by_name
from ..models.schemas import FaceEncoding, Person, RecognizedPerson


def get_face_encoding(image_bytes: bytes) -> Optional[List[float]]:
    """
    Charge les données d'image, trouve le visage et retourne l'encodage de 128D.
    """
    try:
        # Charger l'image en utilisant PIL et la convertir en tableau NumPy (requis par face_recognition)
        image = Image.open(BytesIO(image_bytes))
        # Convertir l'image PIL en un tableau NumPy RGB
        image_np = np.array(image.convert("RGB"))

        # 1. Détecter les emplacements des visages
        face_locations = face_recognition.face_locations(image_np)

        if not face_locations:
            return None  # Aucun visage trouvé

        # 2. Obtenir les encodages (vecteurs de 128 nombres)
        # On ne prend que le premier visage trouvé par image pour simplifier l'ajout.
        encodings = face_recognition.face_encodings(image_np, face_locations)

        # Retourner l'encodage sous forme de liste de floats
        return encodings[0].tolist()

    except Exception as e:
        # Gérer les erreurs de chargement ou de traitement d'image
        print(f"Erreur d'encodage du visage: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de l'image.")


async def process_and_add_person(name: str, files: List[UploadFile]) -> dict:
    """
    Traite plusieurs images pour une personne et les ajoute à la base de données.
    """
    all_encodings: List[FaceEncoding] = []

    # Processus pour garantir la diversité (comme vous l'avez demandé)
    for file in files:
        contents = await file.read()
        encoding_vector = get_face_encoding(contents)

        if encoding_vector:
            all_encodings.append(FaceEncoding(vector=encoding_vector))
        else:
            print(f"Avertissement: Aucun visage trouvé dans le fichier {file.filename}")

    if not all_encodings:
        raise HTTPException(
            status_code=400,
            detail="Aucun visage valide n'a été trouvé dans les images fournies."
        )

    # Préparer la structure de données pour l'insertion
    person_data = {
        "name": name,
        # Convertir les modèles Pydantic en dictionnaires pour MongoDB
        "encodings": [e.dict() for e in all_encodings]
    }

    inserted_person = await add_person_to_db(person_data)
    return {"name": inserted_person['name'], "count": len(all_encodings), "id": inserted_person['id']}

async def recognize_face_from_db(image_bytes: bytes) -> RecognizedPerson:
    """
    Encode un visage inconnu et le compare avec les encodages de la base de données.
    """

    # 1. Obtenir l'encodage du visage à identifier
    unknown_encoding_list = get_face_encoding(image_bytes)

    if not unknown_encoding_list:
        return RecognizedPerson(name="Inconnu", is_recognized=False)

    # Convertir en tableau NumPy (requis par face_recognition pour la comparaison)
    unknown_encoding = np.array(unknown_encoding_list)

    # 2. Charger toutes les données de la base de données
    known_people = await get_all_encodings()

    # Préparer les listes pour la comparaison
    known_face_encodings = []
    known_face_names = []

    for person in known_people:
        name = person['name']

        # Parcourir tous les encodages (images) stockés pour cette personne
        for encoding_data in person['encodings']:
            # L'encodage est stocké en DB comme une liste, le convertir en NumPy array
            face_vector = np.array(encoding_data['vector'])

            known_face_encodings.append(face_vector)
            known_face_names.append(name)  # Garder le nom correspondant à l'encodage

    if not known_face_encodings:
        return RecognizedPerson(name="Inconnu", is_recognized=False)  # Aucune donnée en DB

    # 3. Comparaison (le cœur de la reconnaissance)
    # `compare_faces` utilise la distance euclidienne. La tolérance par défaut est 0.6.
    matches = face_recognition.compare_faces(
        known_face_encodings,
        unknown_encoding,
        tolerance=0.6  # Vous pouvez ajuster cette tolérance (0.6 est standard)
    )

    # 4. Identifier le nom si un match est trouvé
    recognized_name = "Inconnu"
    is_recognized = False

    if True in matches:
        first_match_index = matches.index(True)
        recognized_name = known_face_names[first_match_index]
        is_recognized = True

        # OPTIONNEL: Si plusieurs encodages d'une même personne correspondent, vous pouvez affiner
        # en calculant la distance moyenne ou la plus faible pour confirmer l'identité.

    return RecognizedPerson(name=recognized_name, is_recognized=is_recognized)