# API de prédiction de sentiment des tweets – Projet Air Paradis

## 🎯 Objectif du projet
Ce projet est un prototype développé par **MIC (Marketing Intelligence Consulting)** pour la compagnie aérienne **Air Paradis**.  
Il permet de **prédire le sentiment d’un tweet** (positif, négatif, neutre) afin d’anticiper les **bad buzz** sur les réseaux sociaux.  

Le modèle est :
- Entraîné à partir de données publiques [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140).
- Sauvegardé et suivi avec **MLflow**.
- Déployé via une **API FastAPI** hébergée sur un cloud gratuit (ex. Azure WebApp / Heroku / PythonAnywhere).
- Monitoré avec **Azure Application Insights** pour tracer les prédictions et déclencher des alertes en cas d’erreurs répétées.

## 🗂️ Organisation du dossier

api/
├── .github/workflows/deploy.yml # Pipeline CI/CD (déploiement continu)
├── Procfile # Lancement de l’API (Heroku / Cloud)
├── README.md # Documentation projet (ce fichier)
├── api.py # Code principal de l’API FastAPI
├── mlflow_model/ # Modèle sauvegardé par MLflow
│ ├── model.pkl
│ ├── MLmodel
│ ├── conda.yaml
│ ├── python_env.yaml
│ └── requirements.txt
├── poetry.lock # Verrouillage des versions (Poetry)
├── pyproject.toml # Déclaration des dépendances (Poetry)
├── requirements.txt # Liste des dépendances (pour déploiement cloud)
├── runtime.txt # Version Python (Heroku)
├── script/
│ └── download_data.py # Script de téléchargement des données Kaggle
├── test_api_negatif.py # Test prédiction tweet négatif
├── test_api_positif.py # Test prédiction tweet positif
└── tests/
└── test_app.py # Tests unitaires de l’API 


## 🚀 Fonctionnalités de l’API
- **GET /** : Vérifie que l’API fonctionne.  
- **GET /health** : Vérifie l’état de santé de l’API.  
- **GET /data-status** : Vérifie la disponibilité des données.  
- **GET /system-status** : Retourne l’état du système (modèle, données, env).  
- **GET /tweets** : Retourne un échantillon de tweets.  
- **POST /predict** : Prédit le sentiment d’un tweet donné.  

## 🧰 Packages principaux
- **FastAPI** / **Uvicorn** → API web.  
- **MLflow** → suivi et déploiement du modèle.  
- **Pandas, Scikit-learn** → traitement des données et entraînement.  
- **Streamlit** → interface utilisateur.  
- **Opencensus** + **Azure Monitor** → monitoring et alertes.  

## ⚙️ Installation
### Avec Poetry
```bash
poetry install
poetry run uvicorn api:app --reload
poetry run streamlit run app_streamlit.py
