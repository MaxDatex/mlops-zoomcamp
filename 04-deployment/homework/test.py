import requests

data = {
    "year": 2023,
    "month": 5,
}

url = 'http://localhost:9696/predict'
responce = requests.post(url, json=data)
print(responce.json())