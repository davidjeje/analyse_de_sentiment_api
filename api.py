from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import mlflow
import pandas as pd
import os
from typing import List
import logging

# Azure Application Insights
from opencensus.ext.azure.log_exporter import AzureLogHandler

# Import du script de téléchargement des données Kaggle
from scripts.download_data import download_sentiment140_data

# --------------------------
# CONFIGURATION LOGGING
# --------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INSTRUMENTATION_KEY = os.getenv(
    "APPLICATIONINSIGHTS_CONNECTION_STRING",
    "dc668c66-e558-48aa-aedd-845404a47a18;IngestionEndpoint=https://canadacentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://canadacentral.livediagnostics.monitor.azure.com/;ApplicationId=7f4f2e23-49d7-4d86-82e9-f80804203f34"
)
logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))

# --------------------------
# CHEMINS DES DONNÉES ET DU MODÈLE
# --------------------------
MODEL_DIR = "./mlflow_model"
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "model")
DATA_PATH = os.path.join("data", "training.1600000.processed.noemoticon.csv")

# --------------------------
# TÉLÉCHARGEMENT AUTOMATIQUE
# --------------------------
# Télécharger le modèle MLflow si absent
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

# Télécharger les données Kaggle si absentes
if not os.path.exists(DATA_PATH):
    try:
        logger.info("Téléchargement des données Sentiment140 depuis Kaggle...")
        download_sentiment140_data()
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
    """Vérifie que les données sont disponibles"""
    if not os.path.exists(DATA_PATH):
        logger.warning(f"❌ Fichier de données non trouvé : {DATA_PATH}")
        return False
    logger.info(f"✅ Fichier de données trouvé : {DATA_PATH}")
    return True

if not check_data_availability():
    logger.warning("⚠️  API démarrée sans données - certains endpoints ne fonctionneront pas")

# --------------------------
# FASTAPI APP
# --------------------------
app = FastAPI()

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

label_mapping = {
    0: "negative",
    2: "neutral",
    4: "positive"
}

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
        return {
            "data_available": True,
            "data_path": DATA_PATH,
            "file_size_mb": round(file_size / (1024*1024), 1)
        }
    else:
        return {
            "data_available": False,
            "message": "Les données seront téléchargées automatiquement au démarrage"
        }

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
    if os.path.exists(data_dir):
        status["data_directory_files"] = os.listdir(data_dir)
    else:
        status["data_directory_files"] = "Directory does not exist"
    
    return status

@app.get("/tweets", response_model=TweetsOut)
def get_tweets(sample_frac: float = 0.15):
    if not os.path.exists(DATA_PATH):
        logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
        raise HTTPException(
            status_code=404, 
            detail=f"Fichier de données non trouvé : {DATA_PATH}"
        )

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
    confidence = 1.0  # optionnel, MLflow pyfunc ne renvoie pas toujours la probabilité
    sentiment = label_mapping.get(pred_label, "unknown")
    return PredictionOut(sentiment=sentiment, confidence=confidence)



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import mlflow.pyfunc
# import mlflow
# import pandas as pd
# import os
# from typing import List

# # Azure Application Insights
# from opencensus.ext.azure.log_exporter import AzureLogHandler
# import logging

# # Config logger pour Azure Application Insights
# logger = logging.getLogger(__name__)
# INSTRUMENTATION_KEY = os.getenv(
#     "APPLICATIONINSIGHTS_CONNECTION_STRING",
#     "dc668c66-e558-48aa-aedd-845404a47a18;IngestionEndpoint=https://canadacentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://canadacentral.livediagnostics.monitor.azure.com/;ApplicationId=7f4f2e23-49d7-4d86-82e9-f80804203f34"
# )
# logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))
# logger.setLevel(logging.INFO)

# app = FastAPI()

# # L'URI que tu as donné pointe vers un run stocké sur ton tracking server o
# model_uri = "runs:/62c9722eb896400dabe73d9302cddea7/model"

# # Exporter le modèle en local (si pas déjà téléchargé)
# if not os.path.exists("./mlflow_model"):
#     try:
#         logger.info("Téléchargement du modèle MLflow en cours...")
#         mlflow.artifacts.download_artifacts(
#             artifact_uri=model_uri,
#             dst_path="./mlflow_model"
#         )
#         logger.info("✅ Modèle téléchargé avec succès")
#     except Exception as e:
#         logger.error(f"❌ Erreur lors du téléchargement du modèle: {e}")
#         raise

# # 🔥 CORRECTION : Charger le modèle ICI, avant les endpoints
# try:
#     local_model_uri = "./mlflow_model/model"  # Chemin corrigé avec /model
#     model = mlflow.pyfunc.load_model(local_model_uri)
#     logger.info(f"✅ Modèle chargé depuis : {local_model_uri}")
# except Exception as e:
#     logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
#     raise

# # 📌 CORRECTION : Chemin relatif depuis /app/
# DATA_PATH = os.path.join("data", "training.1600000.processed.noemoticon.csv")

# # 🚨 VÉRIFICATION : Le fichier doit être téléchargé via le script séparé
# def check_data_availability():
#     """
#     Vérifie que les données sont disponibles.
#     Si pas disponibles, suggère d'exécuter le script de téléchargement.
#     """
#     if not os.path.exists(DATA_PATH):
#         error_msg = (
#             f"❌ Fichier de données non trouvé : {DATA_PATH}\n"
#             f"💡 Le script startup.sh devrait télécharger les données automatiquement"
#         )
#         logger.error(error_msg)
#         return False
#     logger.info(f"✅ Fichier de données trouvé : {DATA_PATH}")
#     return True

# # Vérification au démarrage
# if not check_data_availability():
#     logger.warning("⚠️  API démarrée sans données - certains endpoints ne fonctionneront pas")

# class TweetIn(BaseModel):
#     text: str

# class PredictionOut(BaseModel):
#     sentiment: str
#     confidence: float

# class TweetsOut(BaseModel):
#     tweets: List[str]

# label_mapping = {
#     0: "negative",
#     2: "neutral",
#     4: "positive"
# }

# @app.get("/")
# def read_root():
#     return {"message": "API FastAPI OK"}

# @app.get("/health")
# def health_check():
#     logger.info("Health check OK")
#     return {"status": "ok"}

# @app.get("/data-status")
# def data_status():
#     """
#     Endpoint pour vérifier si les données sont disponibles
#     """
#     data_available = os.path.exists(DATA_PATH)
#     if data_available:
#         file_size = os.path.getsize(DATA_PATH)
#         return {
#             "data_available": True,
#             "data_path": DATA_PATH,
#             "file_size_mb": round(file_size / (1024*1024), 1)
#         }
#     else:
#         return {
#             "data_available": False,
#             "message": "Les données seront téléchargées automatiquement au démarrage"
#         }

# @app.get("/system-status")
# def system_status():
#     """
#     Endpoint de diagnostic complet du système
#     """
#     data_available = os.path.exists(DATA_PATH)
#     model_loaded = 'model' in globals() and model is not None
    
#     status = {
#         "api_status": "running",
#         "data_available": data_available,
#         "model_loaded": model_loaded,
#         "environment": "production" if os.getenv("WEBSITE_SITE_NAME") else "development",
#         "current_directory": os.getcwd()
#     }
    
#     if data_available:
#         file_size = os.path.getsize(DATA_PATH)
#         status["data_size_mb"] = round(file_size / (1024*1024), 1)
    
#     # Vérifier les variables d'environnement critiques
#     status["kaggle_configured"] = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    
#     # Lister les fichiers du répertoire data
#     data_dir = "data"
#     if os.path.exists(data_dir):
#         status["data_directory_files"] = os.listdir(data_dir)
#     else:
#         status["data_directory_files"] = "Directory does not exist"
    
#     return status

# @app.get("/tweets", response_model=TweetsOut)
# def get_tweets(sample_frac: float = 0.15):
#     if not os.path.exists(DATA_PATH):
#         logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
#         raise HTTPException(
#             status_code=404, 
#             detail=(
#                 f"Fichier de données non trouvé : {DATA_PATH}. "
#                 f"Vérifiez que le script de téléchargement s'est bien exécuté au démarrage."
#             )
#         )

#     col_names = ["target", "ids", "date", "flag", "user", "text"]
#     df = pd.read_csv(DATA_PATH, encoding="latin-1", names=col_names, header=None)
#     df_text = df["text"].sample(frac=sample_frac, random_state=42).reset_index(drop=True)

#     logger.info(f"Retourne {len(df_text)} tweets en échantillon")
#     return TweetsOut(tweets=df_text.tolist())

# @app.post("/predict", response_model=PredictionOut)
# def predict_sentiment(tweet: TweetIn):
#     text = [tweet.text]

#     pred = model.predict(text)
#     pred_label = int(pred[0])


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import mlflow.pyfunc
# import mlflow
# import pandas as pd
# import os
# from typing import List

# # Azure Application Insights
# from opencensus.ext.azure.log_exporter import AzureLogHandler
# import logging

# # Config logger pour Azure Application Insights
# logger = logging.getLogger(__name__)
# INSTRUMENTATION_KEY = os.getenv(
#     "APPLICATIONINSIGHTS_CONNECTION_STRING",
#     "dc668c66-e558-48aa-aedd-845404a47a18;IngestionEndpoint=https://canadacentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://canadacentral.livediagnostics.monitor.azure.com/;ApplicationId=7f4f2e23-49d7-4d86-82e9-f80804203f34"
# )
# logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))
# logger.setLevel(logging.INFO)

# app = FastAPI()

# # L'URI que tu as donné pointe vers un run stocké sur ton tracking server
# model_uri = "runs:/62c9722eb896400dabe73d9302cddea7/model"

# # Exporter le modèle en local (si pas déjà téléchargé)
# if not os.path.exists("./mlflow_model"):
#     try:
#         logger.info("Téléchargement du modèle MLflow en cours...")
#         mlflow.artifacts.download_artifacts(
#             artifact_uri=model_uri,
#             dst_path="./mlflow_model"
#         )
#         logger.info("✅ Modèle téléchargé avec succès")
#     except Exception as e:
#         logger.error(f"❌ Erreur lors du téléchargement du modèle: {e}")
#         raise

# # 🔥 CORRECTION : Charger le modèle ICI, avant les endpoints
# try:
#     local_model_uri = "./mlflow_model/model"  # Chemin corrigé avec /model
#     model = mlflow.pyfunc.load_model(local_model_uri)
#     logger.info(f"✅ Modèle chargé depuis : {local_model_uri}")
# except Exception as e:
#     logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
#     raise

# # 📌 Définition du chemin vers le fichier CSV
# DATA_PATH = os.path.join("app", "data", "training.1600000.processed.noemoticon.csv")

# # 🚨 VÉRIFICATION : Le fichier doit être téléchargé via le script séparé
# def check_data_availability():
#     """
#     Vérifie que les données sont disponibles.
#     Si pas disponibles, suggère d'exécuter le script de téléchargement.
#     """
#     if not os.path.exists(DATA_PATH):
#         error_msg = (
#             f"❌ Fichier de données non trouvé : {DATA_PATH}\n"
#             f"💡 Veuillez exécuter le script de téléchargement :\n"
#             f"   poetry run python scripts/download_data.py\n"
#             f"   ou\n"
#             f"   python scripts/download_data.py"
#         )
#         logger.error(error_msg)
#         return False
#     logger.info(f"✅ Fichier de données trouvé : {DATA_PATH}")
#     return True

# # Vérification au démarrage
# if not check_data_availability():
#     logger.warning("⚠️  API démarrée sans données - certains endpoints ne fonctionneront pas")

# class TweetIn(BaseModel):
#     text: str

# class PredictionOut(BaseModel):
#     sentiment: str
#     confidence: float

# class TweetsOut(BaseModel):
#     tweets: List[str]

# label_mapping = {
#     0: "negative",
#     2: "neutral",
#     4: "positive"
# }

# @app.get("/")
# def read_root():
#     return {"message": "API FastAPI OK"}

# @app.get("/health")
# def health_check():
#     logger.info("Health check OK")
#     return {"status": "ok"}

# @app.get("/data-status")
# def data_status():
#     """
#     Endpoint pour vérifier si les données sont disponibles
#     """
#     data_available = os.path.exists(DATA_PATH)
#     if data_available:
#         file_size = os.path.getsize(DATA_PATH)
#         return {
#             "data_available": True,
#             "data_path": DATA_PATH,
#             "file_size_mb": round(file_size / (1024*1024), 1)
#         }
#     else:
#         return {
#             "data_available": False,
#             "message": "Exécutez 'python scripts/download_data.py' pour télécharger les données"
#         }

# @app.get("/tweets", response_model=TweetsOut)
# def get_tweets(sample_frac: float = 0.15):
#     if not os.path.exists(DATA_PATH):
#         logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
#         raise HTTPException(
#             status_code=404, 
#             detail=(
#                 f"Fichier de données non trouvé : {DATA_PATH}. "
#                 f"Exécutez 'python scripts/download_data.py' pour télécharger les données."
#             )
#         )

#     col_names = ["target", "ids", "date", "flag", "user", "text"]
#     df = pd.read_csv(DATA_PATH, encoding="latin-1", names=col_names, header=None)
#     df_text = df["text"].sample(frac=sample_frac, random_state=42).reset_index(drop=True)

#     logger.info(f"Retourne {len(df_text)} tweets en échantillon")
#     return TweetsOut(tweets=df_text.tolist())

# @app.post("/predict", response_model=PredictionOut)
# def predict_sentiment(tweet: TweetIn):
#     text = [tweet.text]

#     pred = model.predict(text)
#     pred_label = int(pred[0])

#     sentiment_label = label_mapping.get(pred_label, "unknown")
#     confidence = 1.0

#     logger.info(f"Prediction réalisée: texte='{tweet.text[:50]}...', sentiment={sentiment_label}")
#     return PredictionOut(sentiment=sentiment_label, confidence=confidence)

# # AJOUTEZ CETTE PARTIE À LA FIN
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import mlflow.pyfunc
# import mlflow
# import pandas as pd
# import os
# from typing import List

# # Azure Application Insights
# from opencensus.ext.azure.log_exporter import AzureLogHandler
# import logging

# # Config logger pour Azure Application Insights
# logger = logging.getLogger(__name__)
# INSTRUMENTATION_KEY = os.getenv(
#     "APPLICATIONINSIGHTS_CONNECTION_STRING",
#     "dc668c66-e558-48aa-aedd-845404a47a18;IngestionEndpoint=https://canadacentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://canadacentral.livediagnostics.monitor.azure.com/;ApplicationId=7f4f2e23-49d7-4d86-82e9-f80804203f34"
# )
# logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))
# logger.setLevel(logging.INFO)

# app = FastAPI()

# # L'URI que tu as donné pointe vers un run stocké sur ton tracking server
# model_uri = "runs:/62c9722eb896400dabe73d9302cddea7/model"

# # Exporter le modèle en local (si pas déjà téléchargé)
# if not os.path.exists("./mlflow_model"):
#     mlflow.artifacts.download_artifacts(
#         artifact_uri=model_uri,
#         dst_path="./mlflow_model"
#     )

# # 📌 Définition du chemin vers le fichier CSV
# DATA_PATH = os.path.join("app", "data", "training.1600000.processed.noemoticon.csv")

# # 🚨 VÉRIFICATION : Le fichier doit être téléchargé via le script séparé
# def check_data_availability():
#     """
#     Vérifie que les données sont disponibles.
#     Si pas disponibles, suggère d'exécuter le script de téléchargement.
#     """
#     if not os.path.exists(DATA_PATH):
#         error_msg = (
#             f"❌ Fichier de données non trouvé : {DATA_PATH}\n"
#             f"💡 Veuillez exécuter le script de téléchargement :\n"
#             f"   poetry run python scripts/download_data.py\n"
#             f"   ou\n"
#             f"   python scripts/download_data.py"
#         )
#         logger.error(error_msg)
#         return False
#     logger.info(f"✅ Fichier de données trouvé : {DATA_PATH}")
#     return True

# # Vérification au démarrage
# if not check_data_availability():
#     logger.warning("⚠️  API démarrée sans données - certains endpoints ne fonctionneront pas")

# class TweetIn(BaseModel):
#     text: str

# class PredictionOut(BaseModel):
#     sentiment: str
#     confidence: float

# class TweetsOut(BaseModel):
#     tweets: List[str]

# label_mapping = {
#     0: "negative",
#     2: "neutral",
#     4: "positive"
# }

# @app.get("/")
# def read_root():
#     return {"message": "API FastAPI OK"}

# @app.get("/health")
# def health_check():
#     logger.info("Health check OK")
#     return {"status": "ok"}

# @app.get("/data-status")
# def data_status():
#     """
#     Endpoint pour vérifier si les données sont disponibles
#     """
#     data_available = os.path.exists(DATA_PATH)
#     if data_available:
#         file_size = os.path.getsize(DATA_PATH)
#         return {
#             "data_available": True,
#             "data_path": DATA_PATH,
#             "file_size_mb": round(file_size / (1024*1024), 1)
#         }
#     else:
#         return {
#             "data_available": False,
#             "message": "Exécutez 'python scripts/download_data.py' pour télécharger les données"
#         }

# @app.get("/tweets", response_model=TweetsOut)
# def get_tweets(sample_frac: float = 0.15):
#     if not os.path.exists(DATA_PATH):
#         logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
#         raise HTTPException(
#             status_code=404, 
#             detail=(
#                 f"Fichier de données non trouvé : {DATA_PATH}. "
#                 f"Exécutez 'python scripts/download_data.py' pour télécharger les données."
#             )
#         )

#     col_names = ["target", "ids", "date", "flag", "user", "text"]
#     df = pd.read_csv(DATA_PATH, encoding="latin-1", names=col_names, header=None)
#     df_text = df["text"].sample(frac=sample_frac, random_state=42).reset_index(drop=True)

#     logger.info(f"Retourne {len(df_text)} tweets en échantillon")
#     return TweetsOut(tweets=df_text.tolist())

# @app.post("/predict", response_model=PredictionOut)
# def predict_sentiment(tweet: TweetIn):
#     text = [tweet.text]

#     pred = model.predict(text)
#     pred_label = int(pred[0])

#     sentiment_label = label_mapping.get(pred_label, "unknown")
#     confidence = 1.0

#     logger.info(f"Prediction réalisée: texte='{tweet.text[:50]}...', sentiment={sentiment_label}")
#     return PredictionOut(sentiment=sentiment_label, confidence=confidence)

# # Tu peux ensuite le recharger depuis le chemin local
# local_model_uri = "./mlflow_model"
# model = mlflow.pyfunc.load_model(local_model_uri)

# # AJOUTEZ CETTE PARTIE À LA FIN
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import mlflow.pyfunc
# import mlflow
# import pandas as pd
# import os
# from typing import List

# # Azure Application Insights
# from opencensus.ext.azure.log_exporter import AzureLogHandler
# import logging

# # Kaggle API pour télécharger les données
# from kaggle.api.kaggle_api_extended import KaggleApi

# # Config logger pour Azure Application Insights
# logger = logging.getLogger(__name__)
# INSTRUMENTATION_KEY = os.getenv(
#     "APPLICATIONINSIGHTS_CONNECTION_STRING",
#     "dc668c66-e558-48aa-aedd-845404a47a18;IngestionEndpoint=https://canadacentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://canadacentral.livediagnostics.monitor.azure.com/;ApplicationId=7f4f2e23-49d7-4d86-82e9-f80804203f34"
# )
# logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))
# logger.setLevel(logging.INFO)

# app = FastAPI()

# # L’URI que tu as donné pointe vers un run stocké sur ton tracking server
# model_uri = "runs:/62c9722eb896400dabe73d9302cddea7/model"

# # Exporter le modèle en local (si pas déjà téléchargé)
# if not os.path.exists("./mlflow_model"):
#     mlflow.artifacts.download_artifacts(
#         artifact_uri=model_uri,
#         dst_path="./mlflow_model"
#     )

# # 📌 Définition du chemin vers le fichier CSV
# DATA_PATH = os.path.join("app", "data", "training.1600000.processed.noemoticon.csv")

# # 📥 Télécharger depuis Kaggle si le fichier n'existe pas
# if not os.path.exists(DATA_PATH):
#     logger.info(f"Fichier de données non trouvé à {DATA_PATH}, téléchargement depuis Kaggle...")
#     os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
#     api = KaggleApi()
#     api.authenticate()
#     api.dataset_download_file(
#         "kazanova/sentiment140",
#         file_name="training.1600000.processed.noemoticon.csv",
#         path=os.path.dirname(DATA_PATH)
#     )
#     logger.info("Téléchargement terminé ✅")

# class TweetIn(BaseModel):
#     text: str

# class PredictionOut(BaseModel):
#     sentiment: str
#     confidence: float

# class TweetsOut(BaseModel):
#     tweets: List[str]

# label_mapping = {
#     0: "negative",
#     2: "neutral",
#     4: "positive"
# }

# @app.get("/")
# def read_root():
#     return {"message": "API FastAPI OK"}

# @app.get("/health")
# def health_check():
#     logger.info("Health check OK")
#     return {"status": "ok"}

# @app.get("/tweets", response_model=TweetsOut)
# def get_tweets(sample_frac: float = 0.15):
#     if not os.path.exists(DATA_PATH):
#         logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
#         raise HTTPException(status_code=404, detail=f"Fichier de données non trouvé à {DATA_PATH}")

#     col_names = ["target", "ids", "date", "flag", "user", "text"]
#     df = pd.read_csv(DATA_PATH, encoding="latin-1", names=col_names, header=None)
#     df_text = df["text"].sample(frac=sample_frac, random_state=42).reset_index(drop=True)

#     logger.info(f"Retourne {len(df_text)} tweets en échantillon")
#     return TweetsOut(tweets=df_text.tolist())

# @app.post("/predict", response_model=PredictionOut)
# def predict_sentiment(tweet: TweetIn):
#     text = [tweet.text]

#     pred = model.predict(text)
#     pred_label = int(pred[0])

#     sentiment_label = label_mapping.get(pred_label, "unknown")
#     confidence = 1.0

#     logger.info(f"Prediction réalisée: texte='{tweet.text[:50]}...', sentiment={sentiment_label}")
#     return PredictionOut(sentiment=sentiment_label, confidence=confidence)

# # Tu peux ensuite le recharger depuis le chemin local
# local_model_uri = "./mlflow_model"
# model = mlflow.pyfunc.load_model(local_model_uri)

# # AJOUTEZ CETTE PARTIE À LA FIN
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import mlflow.pyfunc
# import mlflow
# import pandas as pd
# import os
# from typing import List

# # Azure Application Insights
# from opencensus.ext.azure.log_exporter import AzureLogHandler
# import logging

# # Config logger pour Azure Application Insights
# logger = logging.getLogger(__name__)
# # Remplace "TON_INSTRUMENTATION_KEY" par ta vraie clé dans les variables d'env ou secrets
# INSTRUMENTATION_KEY = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "dc668c66-e558-48aa-aedd-845404a47a18;IngestionEndpoint=https://canadacentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://canadacentral.livediagnostics.monitor.azure.com/;ApplicationId=7f4f2e23-49d7-4d86-82e9-f80804203f34")
# logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))
# logger.setLevel(logging.INFO)

# app = FastAPI()

# # L’URI que tu as donné pointe vers un run stocké sur ton tracking server
# model_uri = "runs:/62c9722eb896400dabe73d9302cddea7/model"

# # Exporter le modèle en local (si pas déjà téléchargé)
# if not os.path.exists("./mlflow_model"):
#     mlflow.artifacts.download_artifacts(
#         artifact_uri=model_uri,
#         dst_path="./mlflow_model"
#     )

# DATA_PATH = "data/training.1600000.processed.noemoticon.csv"

# class TweetIn(BaseModel):
#     text: str

# class PredictionOut(BaseModel):
#     sentiment: str
#     confidence: float

# class TweetsOut(BaseModel):
#     tweets: List[str]

# label_mapping = {
#     0: "negative",
#     2: "neutral",
#     4: "positive"
# }

# @app.get("/")
# def read_root():
#     return {"message": "API FastAPI OK"}

# @app.get("/health")
# def health_check():
#     logger.info("Health check OK")
#     return {"status": "ok"}

# @app.get("/tweets", response_model=TweetsOut)
# def get_tweets(sample_frac: float = 0.15):
#     if not os.path.exists(DATA_PATH):
#         logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
#         raise HTTPException(status_code=404, detail=f"Fichier de données non trouvé à {DATA_PATH}")

#     col_names = ["target", "ids", "date", "flag", "user", "text"]
#     df = pd.read_csv(DATA_PATH, encoding="latin-1", names=col_names, header=None)
#     df_text = df["text"].sample(frac=sample_frac, random_state=42).reset_index(drop=True)

#     logger.info(f"Retourne {len(df_text)} tweets en échantillon")
#     return TweetsOut(tweets=df_text.tolist())

# @app.post("/predict", response_model=PredictionOut)
# def predict_sentiment(tweet: TweetIn):
#     text = [tweet.text]

#     pred = model.predict(text)
#     pred_label = int(pred[0])

#     sentiment_label = label_mapping.get(pred_label, "unknown")
#     confidence = 1.0

#     logger.info(f"Prediction réalisée: texte='{tweet.text[:50]}...', sentiment={sentiment_label}")
#     return PredictionOut(sentiment=sentiment_label, confidence=confidence)

# # Tu peux ensuite le recharger depuis le chemin local
# local_model_uri = "./mlflow_model"
# model = mlflow.pyfunc.load_model(local_model_uri)



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import mlflow.pyfunc
# import pandas as pd
# import os
# from typing import List

# # Azure Application Insights
# from opencensus.ext.azure.log_exporter import AzureLogHandler
# import logging

# # Config logger pour Azure Application Insights
# logger = logging.getLogger(__name__)
# # Remplace "TON_INSTRUMENTATION_KEY" par ta vraie clé dans les variables d'env ou secrets
# INSTRUMENTATION_KEY = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "dc668c66-e558-48aa-aedd-845404a47a18;IngestionEndpoint=https://canadacentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://canadacentral.livediagnostics.monitor.azure.com/;ApplicationId=7f4f2e23-49d7-4d86-82e9-f80804203f34")
# logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))
# logger.setLevel(logging.INFO)

# app = FastAPI()

# # Charge le modèle MLflow enregistré (au démarrage)
# model_uri = "runs:/62c9722eb896400dabe73d9302cddea7/model"
# model = mlflow.pyfunc.load_model(model_uri)

# DATA_PATH = "data/training.1600000.processed.noemoticon.csv"

# class TweetIn(BaseModel):
#     text: str

# class PredictionOut(BaseModel):
#     sentiment: str
#     confidence: float

# class TweetsOut(BaseModel):
#     tweets: List[str]

# label_mapping = {
#     0: "negative",
#     2: "neutral",
#     4: "positive"
# }

# @app.get("/")
# def read_root():
#     return {"message": "API FastAPI OK"}

# @app.get("/health")
# def health_check():
#     logger.info("Health check OK")
#     return {"status": "ok"}

# @app.get("/tweets", response_model=TweetsOut)
# def get_tweets(sample_frac: float = 0.15):
#     if not os.path.exists(DATA_PATH):
#         logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
#         raise HTTPException(status_code=404, detail=f"Fichier de données non trouvé à {DATA_PATH}")

#     col_names = ["target", "ids", "date", "flag", "user", "text"]
#     df = pd.read_csv(DATA_PATH, encoding="latin-1", names=col_names, header=None)
#     df_text = df["text"].sample(frac=sample_frac, random_state=42).reset_index(drop=True)

#     logger.info(f"Retourne {len(df_text)} tweets en échantillon")
#     return TweetsOut(tweets=df_text.tolist())

# @app.post("/predict", response_model=PredictionOut)
# def predict_sentiment(tweet: TweetIn):
#     text = [tweet.text]

#     pred = model.predict(text)
#     pred_label = int(pred[0])

#     sentiment_label = label_mapping.get(pred_label, "unknown")
#     confidence = 1.0

#     logger.info(f"Prediction réalisée: texte='{tweet.text[:50]}...', sentiment={sentiment_label}")
#     return PredictionOut(sentiment=sentiment_label, confidence=confidence)
