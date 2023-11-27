import requests
from time import sleep


url = "http://localhost:9696/predict"

client = {"job": "retired", "duration": 445, "poutcome": "success"}

while True:
    sleep(0.001)
    response = requests.post(url, json=client).json()
    print(response)

