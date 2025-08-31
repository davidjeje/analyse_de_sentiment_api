import sys
import os

# Ajouter le dossier courant à sys.path pour que Python trouve api.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API FastAPI OK"}
