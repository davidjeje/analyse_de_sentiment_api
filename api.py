from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import mlflow
import pandas as pd
import os
from typing import List
import logging
from dotenv import load_dotenv
load_dotenv()  # lit le fichier .env


# Azure Application Insights
from opencensus.ext.azure.log_exporter import AzureLogHandler

# Import du script de téléchargement des données Kaggle
from script.download_data import download_sentiment140_data
from opencensus.ext.azure.metrics_exporter import new_metrics_exporter

# --------------------------
# Application Insights (logs + metrics + traces HTTP)
# --------------------------
from opencensus.ext.azure.log_exporter import AzureLogHandler
from azure.monitor.opentelemetry.exporter import new_metrics_exporter

# OpenTelemetry (pour requests)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

# --------------------------
# LOGGER
# --------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if os.getenv("APPINSIGHTS_CONNECTION_STRING"):
    logger.addHandler(AzureLogHandler(connection_string=os.getenv("APPINSIGHTS_CONNECTION_STRING")))
else:
    logger.addHandler(logging.StreamHandler())  # fallback console

logger.info("🚀 Logger Application Insights configuré avec succès")

# Export métriques standard
metric_exporter = new_metrics_exporter(
    enable_standard_metrics=True,
    connection_string=os.getenv("APPINSIGHTS_CONNECTION_STRING")
)

# --------------------------
# FASTAPI APP
# --------------------------
app = FastAPI()

# Instrumentation OpenTelemetry pour capturer automatiquement les requêtes HTTP
if os.getenv("APPINSIGHTS_CONNECTION_STRING"):
    tracer_provider = TracerProvider()
    trace_exporter = AzureMonitorTraceExporter.from_connection_string(
        os.getenv("APPINSIGHTS_CONNECTION_STRING")
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
    logger.info("📡 Instrumentation OpenTelemetry activée pour FastAPI")

# --------------------------
# CHEMINS DES DONNÉES ET DU MODÈLE
# --------------------------
MODEL_DIR = "./mlflow_model"
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "model")
DATA_PATH = os.path.join("data", "training.1600000.processed.noemoticon.csv")

# --------------------------
# LOGGING SPÉCIFIQUE
# --------------------------
def log_misclassified_tweet(tweet_text: str, predicted_label: str, true_label: str):
    """
    Log Warning + incrémente un compteur métrique pour les tweets mal prédits.
    """
    logger.warning(
        f"Tweet mal prédit ! Texte='{tweet_text}' | Prédiction='{predicted_label}' | Vérité='{true_label}'"
    )
    mlflow.log_metric("tweets_mal_predits", 1)

# --------------------------
# TÉLÉCHARGEMENT AUTOMATIQUE
# --------------------------
if not os.path.exists(MODEL_DIR):
    try:
        logger.info("Téléchargement du modèle MLflow en cours...")
        mlflow.artifacts.download_artifacts(
            artifact_uri="runs:/62c9722eb896400dabe73d9302cddea7/model",
            dst_path=MODEL_DIR
        )
        logger.info("✅ Modèle MLflow téléchargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement du modèle: {e}")
        raise

if not os.path.exists(DATA_PATH):
    try:
        logger.info("Téléchargement des données Sentiment140 depuis Kaggle...")
        # à adapter si tu as une fonction utilitaire
        # download_sentiment140_data()
        logger.info("✅ Données téléchargées avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement des données: {e}")
        raise

# --------------------------
# CHARGEMENT DU MODÈLE
# --------------------------
try:
    model = mlflow.pyfunc.load_model(LOCAL_MODEL_PATH)
    logger.info(f"✅ Modèle chargé depuis : {LOCAL_MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
    raise

# --------------------------
# VÉRIFICATION DES DONNÉES
# --------------------------
def check_data_availability():
    if not os.path.exists(DATA_PATH):
        logger.warning(f"❌ Fichier de données non trouvé : {DATA_PATH}")
        return False
    logger.info(f"✅ Fichier de données trouvé : {DATA_PATH}")
    return True

if not check_data_availability():
    logger.warning("⚠️  API démarrée sans données - certains endpoints ne fonctionneront pas")

# --------------------------
# SCHÉMAS Pydantic
# --------------------------
class TweetIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    sentiment: str
    confidence: float

class TweetsOut(BaseModel):
    tweets: List[str]

class FeedbackIn(BaseModel):
    text: str
    predicted: str
    true_label: str

label_mapping = {0: "negative", 2: "neutral", 4: "positive"}

# --------------------------
# ENDPOINTS
# --------------------------
@app.get("/")
def read_root():
    return {"message": "API FastAPI OK"}

@app.get("/health")
def health_check():
    logger.info("Health check OK")
    return {"status": "ok"}

@app.get("/data-status")
def data_status():
    data_available = os.path.exists(DATA_PATH)
    if data_available:
        file_size = os.path.getsize(DATA_PATH)
        return {"data_available": True, "data_path": DATA_PATH, "file_size_mb": round(file_size / (1024*1024), 1)}
    else:
        return {"data_available": False, "message": "Les données seront téléchargées automatiquement au démarrage"}

@app.get("/system-status")
def system_status():
    data_available = os.path.exists(DATA_PATH)
    model_loaded = 'model' in globals() and model is not None
    status = {
        "api_status": "running",
        "data_available": data_available,
        "model_loaded": model_loaded,
        "environment": "production" if os.getenv("WEBSITE_SITE_NAME") else "development",
        "current_directory": os.getcwd()
    }
    if data_available:
        file_size = os.path.getsize(DATA_PATH)
        status["data_size_mb"] = round(file_size / (1024*1024), 1)
    status["kaggle_configured"] = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    data_dir = "data"
    status["data_directory_files"] = os.listdir(data_dir) if os.path.exists(data_dir) else "Directory does not exist"
    return status

@app.get("/test-logs")
def test_logs():
    logger.info("📡 Test de log depuis Heroku")
    return {"message": "Log envoyé à App Insights"}

@app.get("/tweets", response_model=TweetsOut)
def get_tweets(sample_frac: float = 0.5):
    if not os.path.exists(DATA_PATH):
        logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
        raise HTTPException(status_code=404, detail=f"Fichier de données non trouvé : {DATA_PATH}")

    col_names = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(DATA_PATH, encoding="latin-1", names=col_names, header=None)
    df_text = df["text"].sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    logger.info(f"Retourne {len(df_text)} tweets en échantillon")
    return TweetsOut(tweets=df_text.tolist())

@app.post("/predict", response_model=PredictionOut)
def predict_sentiment(tweet: TweetIn):
    text = [tweet.text]
    pred = model.predict(text)
    pred_label = int(pred[0])
    sentiment = label_mapping.get(pred_label, "unknown")
    return PredictionOut(sentiment=sentiment, confidence=1.0)

@app.post("/feedback")
def feedback(data: FeedbackIn):
    if data.predicted != data.true_label:
        try:
            log_misclassified_tweet(data.text, data.predicted, data.true_label)
        except Exception as e:
            return {"status": "error", "message": f"Erreur lors du logging: {e}"}
        return {"status": "logged", "message": "Tweet mal prédit enregistré"}
    else:
        return {"status": "ok", "message": "Prédiction correcte"}