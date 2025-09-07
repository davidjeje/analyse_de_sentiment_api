import streamlit as st
import requests

st.title("Analyse de sentiment des tweets")

# Input du tweet
tweet_text = st.text_area("Entrez votre tweet :")

if st.button("Prédire le sentiment"):
    if tweet_text.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        # Appel à l'API locale
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": tweet_text}
        )
        if response.status_code == 200:
            data = response.json()
            st.success(f"Sentiment : {data['sentiment']} (confiance : {data['confidence']})")
        else:
            st.error(f"Erreur API : {response.status_code}")
