import requests
from discord import Message

url = "http://127.0.0.1:8000/predict"


def predict_toxicity(message: Message, threshold: float = 0.8) -> tuple:
    data = {"input": message.content}
    response = requests.post(url=url, json=data).json()

    positive_labels = []

    for label, prob in response.items():
        if prob >= threshold:
            positive_labels.append(label)

    is_toxic = True if len(positive_labels) > 0 else False

    return (is_toxic, positive_labels)
