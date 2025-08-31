import requests

url = "https://analyse-de-sentiment-api-e8ed38a27042.herokuapp.com/predict"
data = {"text": "Very BAD."}

response = requests.post(url, json=data)
print(response.json())
