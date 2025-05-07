import requests

url = "http://127.0.0.1:8000/predict"
test_json = {"input": "You are a piece of shit."}

response = requests.post(url, json=test_json)
print(response.json())
