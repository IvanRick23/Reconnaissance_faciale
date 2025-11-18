
from fastapi import FastAPI
from dotenv import load_dotenv
from app.controllers import face_routes
from app.db.storage import close_mongo_connection # Importez la fonction de fermeture

load_dotenv()

app = FastAPI(
    title="API de Reconnaissance Faciale 1.0",
    description="Services pour Ajouter, Supprimer et Reconnaître des visages."
)

# Inclure les routes...
app.include_router(face_routes.router, prefix="/api/v1/faces", tags=["Reconnaissance Faciale"])

# --- Gestion des événements de cycle de vie ---

@app.on_event("startup")
async def startup_db_client():
    """Initialise la connexion MongoDB. (Elle est déjà faite au niveau du storage,
    mais on peut ajouter ici des vérifications si besoin)."""
    print("Starting up...")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Ferme la connexion MongoDB proprement."""
    await close_mongo_connection()

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de Reconnaissance Faciale."}