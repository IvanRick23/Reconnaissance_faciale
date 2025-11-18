
from pydantic import BaseModel
from typing import List


class FaceEncoding(BaseModel):
    """Modèle pour un encodage facial (vecteur de 128 flottants)"""
    # L'encodage est un tableau de 128 nombres.
    vector: List[float]


class Person(BaseModel):
    """Modèle pour une personne enregistrée dans la base de données"""
    name: str
    encodings: List[FaceEncoding]  # Une liste d'encodages pour la robustesse


class RecognizedPerson(BaseModel):
    """Modèle pour la réponse de reconnaissance"""
    name: str
    is_recognized: bool