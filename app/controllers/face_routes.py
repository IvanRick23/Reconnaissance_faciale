# reconnaissance_faciale/app/controllers/face_routes.py (Mise à jour)

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import List

from ..models.schemas import RecognizedPerson
# Importation du nouveau service
from ..services.face_service import process_and_add_person, recognize_face_from_db

router = APIRouter()


@router.post("/ajouter")
async def add_person_to_system(
        name: str = Form(...),  # Utiliser Form pour les données textuelles dans un upload multipart
        files: List[UploadFile] = File(...)  # Utiliser List[UploadFile] pour recevoir plusieurs images
):
    """
    Endpoint pour ajouter une nouvelle personne.
    Requiert : un nom (str) et au moins 5 images (files: List[UploadFile])
    """
    if len(files) < 5:  # On met une contrainte minimale pour la robustesse
        raise HTTPException(status_code=400,
                            detail="Veuillez fournir au moins 5 images pour une reconnaissance performantes.")

    # Appel du service qui gère l'encodage et le stockage
    result = await process_and_add_person(name, files)

    return {
        "status": "success",
        "message": f"Personne '{result['name']}' ajoutée avec succès.",
        "encodings_saved": result['count']
    }


@router.post("/reconnaître", response_model=RecognizedPerson)
async def recognize_face(file: UploadFile = File(...)):
    """
    Endpoint pour vérifier si un visage est connu.
    Requiert : une seule image (file: UploadFile) d'un visage à identifier.
    """
    # Lire le contenu du fichier
    contents = await file.read()

    # Appel du service de reconnaissance
    result = await recognize_face_from_db(contents)

    # Si la personne est reconnue, l'alarme n'est pas déclenchée.
    # Si 'Inconnu', l'alarme doit être déclenchée (logique à mettre dans un service IoT/Alarm).

    return result


@router.delete("/supprimer/{name}", status_code=status.HTTP_200_OK)
async def delete_person(name: str):
    """
    Endpoint pour supprimer une personne du système par son nom.
    """
    is_deleted = await delete_person_by_name(name)

    if is_deleted:
        return {"status": "success", "message": f"Personne '{name}' supprimée avec succès."}
    else:
        # Retourner une 404 si la personne n'est pas trouvée pour la suppression
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Personne '{name}' non trouvée dans la base de données."
        )


# Ajouter une route GET pour voir les enregistrements (pour le débogage)
from ..db.storage import get_all_encodings, delete_person_by_name


@router.get("/enregistrements", tags=["Admin/Debug"])
async def list_all_people():
    """Liste toutes les personnes et leurs encodages (pour le débogage)."""
    return await get_all_encodings()