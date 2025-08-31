import sys
import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ajouter le dossier courant à sys.path pour que Python trouve api.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app

client = TestClient(app)


# -----------------------------
# Routes simples : / et /health
# -----------------------------

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API FastAPI OK"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# -----------------------------
# /data-status
# -----------------------------

def test_data_status_without_data(monkeypatch, tmp_path):
    # Forcer DATA_PATH vers un fichier inexistant
    monkeypatch.setattr("api.DATA_PATH", str(tmp_path / "fake.csv"))
    response = client.get("/data-status")
    assert response.status_code == 200
    data = response.json()
    assert data["data_available"] is False


def test_data_status_with_data(monkeypatch, tmp_path):
    # Créer un faux fichier CSV
    fake_file = tmp_path / "fake.csv"
    fake_file.write_text("target,ids,date,flag,user,text\n0,1,2020-01-01,N,user1,hello world")

    monkeypatch.setattr("api.DATA_PATH", str(fake_file))
    response = client.get("/data-status")
    assert response.status_code == 200
    data = response.json()
    assert data["data_available"] is True
    assert "file_size_mb" in data


# -----------------------------
# /system-status
# -----------------------------

def test_system_status_without_data(monkeypatch, tmp_path):
    # Pas de fichier → data_available = False
    monkeypatch.setattr("api.DATA_PATH", str(tmp_path / "fake.csv"))
    response = client.get("/system-status")
    assert response.status_code == 200
    data = response.json()
    assert data["api_status"] == "running"
    assert data["data_available"] is False
    assert "environment" in data


# -----------------------------
# /tweets
# -----------------------------

def test_tweets_without_data(monkeypatch, tmp_path):
    # Pas de fichier CSV → doit renvoyer 404
    monkeypatch.setattr("api.DATA_PATH", str(tmp_path / "fake.csv"))
    response = client.get("/tweets")
    assert response.status_code == 404


def test_tweets_with_data(monkeypatch, tmp_path):
    # Créer un faux fichier CSV
    fake_file = tmp_path / "fake.csv"
    df = pd.DataFrame({
        "target": [0, 1],
        "ids": [1, 2],
        "date": ["2020-01-01", "2020-01-02"],
        "flag": ["N", "N"],
        "user": ["user1", "user2"],
        "text": ["hello", "world"]
    })
    df.to_csv(fake_file, index=False, header=False, encoding="latin-1")

    monkeypatch.setattr("api.DATA_PATH", str(fake_file))
    response = client.get("/tweets?sample_frac=1.0")  # tout récupérer
    assert response.status_code == 200
    data = response.json()
    assert "tweets" in data
    assert len(data["tweets"]) == 2


# -----------------------------
# /predict
# -----------------------------

class DummyModel:
    def predict(self, texts):
        return [1]  # Simule toujours "positive"


def test_predict_with_dummy_model(monkeypatch):
    from api import label_mapping
    monkeypatch.setattr("api.model", DummyModel())
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert data["sentiment"] in list(label_mapping.values()) + ["unknown"]
    assert "confidence" in data
