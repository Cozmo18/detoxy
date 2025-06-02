import requests

response = requests.post(
    "http://0.0.0.0:8000/predict", json={"input": "fuck you bitch"}
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
