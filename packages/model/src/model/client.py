import argparse

import requests

API_URL = "http://127.0.0.1:8000/predict"


def send_generate_request(text: str) -> None:
    response = requests.post(API_URL, json={"input": text})
    if response.status_code == 200:
        print(response.text)
    else:
        print(
            f"Error: Response with status code {response.status_code - {response.text}}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send text to the model server and receive classification results."
    )
    parser.add_argument(
        "--text", required=True, help="Raw text to send to classification model."
    )
    args = parser.parse_args()

    send_generate_request(args.text)
