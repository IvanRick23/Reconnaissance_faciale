
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import HTTPException
import os
from bson import ObjectId

MONGO_DETAILS = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "face_recognition_db"
COLLECTION_NAME = "known_faces"

# Initialisation
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client[DB_NAME]
collection = database[COLLECTION_NAME]

async def close_mongo_connection():
    """Ferme la connexion MongoDB lors de l'arrêt de l'application."""
    client.close()
    print("MongoDB connection closed.")

# --- Fonctions CRUD de base ---

def person_helper(person) -> dict:
    """Aide la fonction find à retourner le bon format (conversion ObjectId -> str)."""
    return {
        "id": str(person["_id"]),
        "name": person["name"],
        "encodings": person["encodings"], # Les encodages restent des listes
    }

async def add_person_to_db(person_data: dict) -> dict:
    """Ajoute une nouvelle personne et ses encodages."""
    existing = await collection.find_one({"name": person_data["name"]})
    if existing:
        # Optimisation: lever l'exception ici pour un retour rapide
        raise HTTPException(status_code=400, detail=f"Cet personne '{person_data['name']}' existe déja.")

    new_person = await collection.insert_one(person_data)
    created_person = await collection.find_one({"_id": new_person.inserted_id})
    # Utiliser le helper pour garantir un format de retour propre
    return person_helper(created_person)

async def get_all_encodings() -> List[dict]:
    """Récupère tous les encodages stockés (pour la reconnaissance)."""
    people = []
    # Optimisation: itération asynchrone pour la performance
    async for person in collection.find():
        people.append(person_helper(person))
    return people

async def delete_person_by_name(name: str) -> bool:
    """Supprime une personne par son nom. Retourne True si une suppression a eu lieu."""
    result = await collection.delete_one({"name": name})
    # Optimisation: vérifier directement le nombre de documents supprimés
    return result.deleted_count > 0