# tests/test_app.py
from fastapi.testclient import TestClient
from api import app  # ou ton fichier principal

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API FastAPI OK"}
