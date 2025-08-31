#!/usr/bin/env python3
"""
Script de téléchargement des données depuis Kaggle
À exécuter depuis /home/site/wwwroot/app/ sur Azure
"""

import os
import sys
import logging
import time

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_kaggle_credentials():
    """Vérifie que les credentials Kaggle sont configurés"""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    
    if not username or not key:
        logger.error("❌ Variables d'environnement Kaggle manquantes:")
        logger.error("   - KAGGLE_USERNAME")
        logger.error("   - KAGGLE_KEY")
        logger.error("💡 Configurez-les dans Azure App Service > Configuration > Application Settings")
        return False
    
    logger.info(f"✅ Credentials Kaggle trouvés pour l'utilisateur: {username}")
    return True

def download_sentiment140_data():
    """
    Télécharge le dataset Sentiment140 depuis Kaggle
    """
    # CORRECTION : Depuis /app/, le chemin est simplement "data"
    data_dir = "data"
    data_path = os.path.join(data_dir, "training.1600000.processed.noemoticon.csv")
    
    logger.info(f"🔍 Répertoire de travail actuel : {os.getcwd()}")
    logger.info(f"🎯 Chemin cible des données : {data_path}")
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path) / (1024*1024)
        logger.info(f"✅ Fichier déjà présent : {data_path} ({file_size:.1f} MB)")
        return data_path
    
    # Vérifier les credentials
    if not check_kaggle_credentials():
        raise Exception("Credentials Kaggle manquants")
    
    # Créer le dossier de données s'il n'existe pas
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"📁 Dossier créé/vérifié : {data_dir}")
    
    try:
        # Initialiser l'API Kaggle
        logger.info("🔑 Authentification Kaggle...")
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        logger.info("✅ Authentification Kaggle réussie")
        
        # Télécharger le dataset avec retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"📥 Téléchargement du dataset Sentiment140 (tentative {attempt + 1}/{max_retries})...")
                
                api.dataset_download_file(
                    "kazanova/sentiment140",
                    file_name="training.1600000.processed.noemoticon.csv",
                    path=data_dir
                )
                
                if os.path.exists(data_path):
                    file_size = os.path.getsize(data_path) / (1024*1024)
                    logger.info(f"✅ Téléchargement terminé : {data_path} ({file_size:.1f} MB)")
                    return data_path
                else:
                    raise Exception("Fichier non créé après téléchargement")
                    
            except Exception as e:
                logger.warning(f"⚠️  Tentative {attempt + 1} échouée : {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Attendre 5s avant retry
                else:
                    raise
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement : {e}")
        raise

def verify_data_integrity():
    """
    Vérifie l'intégrité des données téléchargées
    """
    data_path = os.path.join("data", "training.1600000.processed.noemoticon.csv")
    
    if not os.path.exists(data_path):
        logger.error(f"❌ Fichier non trouvé : {data_path}")
        return False
    
    # Vérifier la taille du fichier
    file_size = os.path.getsize(data_path)
    file_size_mb = file_size / (1024*1024)
    logger.info(f"📊 Taille du fichier : {file_size_mb:.1f} MB")
    
    # Le fichier devrait faire environ 238 MB
    if file_size_mb < 200:
        logger.warning(f"⚠️  Fichier trop petit ({file_size_mb:.1f} MB), téléchargement possiblement incomplet")
        return False
    
    # Vérification basique du contenu
    try:
        import pandas as pd
        col_names = ["target", "ids", "date", "flag", "user", "text"]
        df = pd.read_csv(data_path, encoding="latin-1", names=col_names, header=None, nrows=10)
        
        logger.info(f"✅ Fichier lisible, {len(df)} lignes testées")
        logger.info(f"📈 Colonnes détectées : {list(df.columns)}")
        
        # Vérifier les valeurs target attendues
        unique_target


# #!/usr/bin/env python3
# """
# Script de téléchargement des données depuis Kaggle
# À exécuter AVANT de lancer l'API
# """

# import os
# import logging
# from kaggle.api.kaggle_api_extended import KaggleApi

# # Configuration du logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def download_sentiment140_data():
#     """
#     Télécharge le dataset Sentiment140 depuis Kaggle
#     """
#     # Définition du chemin vers le fichier CSV
#     data_dir = os.path.join("app", "data")
#     data_path = os.path.join(data_dir, "training.1600000.processed.noemoticon.csv")
    
#     # Vérifier si le fichier existe déjà
#     if os.path.exists(data_path):
#         logger.info(f"✅ Fichier déjà présent : {data_path}")
#         return data_path
    
#     # Créer le dossier de données s'il n'existe pas
#     os.makedirs(data_dir, exist_ok=True)
#     logger.info(f"📁 Dossier créé : {data_dir}")
    
#     try:
#         # Initialiser l'API Kaggle
#         logger.info("🔑 Authentification Kaggle...")
#         api = KaggleApi()
#         api.authenticate()
#         logger.info("✅ Authentification Kaggle réussie")
        
#         # Télécharger le dataset
#         logger.info("📥 Téléchargement du dataset Sentiment140...")
#         api.dataset_download_file(
#             "kazanova/sentiment140",
#             file_name="training.1600000.processed.noemoticon.csv",
#             path=data_dir
#         )
        
#         logger.info(f"✅ Téléchargement terminé : {data_path}")
#         return data_path
        
#     except Exception as e:
#         logger.error(f"❌ Erreur lors du téléchargement : {e}")
#         raise

# def verify_data_integrity():
#     """
#     Vérifie l'intégrité des données téléchargées
#     """
#     data_path = os.path.join("app", "data", "training.1600000.processed.noemoticon.csv")
    
#     if not os.path.exists(data_path):
#         logger.error(f"❌ Fichier non trouvé : {data_path}")
#         return False
    
#     # Vérifier la taille du fichier
#     file_size = os.path.getsize(data_path)
#     logger.info(f"📊 Taille du fichier : {file_size / (1024*1024):.1f} MB")
    
#     # Vérification basique du contenu
#     try:
#         import pandas as pd
#         col_names = ["target", "ids", "date", "flag", "user", "text"]
#         df = pd.read_csv(data_path, encoding="latin-1", names=col_names, header=None, nrows=5)
#         logger.info(f"✅ Fichier lisible, colonnes : {list(df.columns)}")
#         logger.info(f"📈 Aperçu des premières lignes :\n{df[['target', 'text']].head()}")
#         return True
#     except Exception as e:
#         logger.error(f"❌ Erreur lors de la vérification : {e}")
#         return False

# if __name__ == "__main__":
#     logger.info("🚀 Démarrage du script de téléchargement des données")
    
#     try:
#         # Télécharger les données
#         data_path = download_sentiment140_data()
        
#         # Vérifier l'intégrité
#         if verify_data_integrity():
#             logger.info("🎉 Données prêtes pour l'API !")
#         else:
#             logger.error("❌ Problème avec les données téléchargées")
            
#     except Exception as e:
#         logger.error(f"💥 Échec du script : {e}")
#         exit(1)