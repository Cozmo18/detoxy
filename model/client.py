import argparse
import os

import requests
from dotenv import load_dotenv

load_dotenv()

# Update this URL to your server's URL if hosted remotely
API_URL = os.environ.get("API_URL")


def send_generate_request(text: str):
    response = requests.post(API_URL, json={"input": text})
    if response.status_code == 200:
        print(response.text)
    else:
        print(
            f"Error: Response with status code {response.status_code - {response.text}}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send text to the model server and receive classification results."
    )
    parser.add_argument(
        "--text", required=True, help="Raw text to send to classification model."
    )
    args = parser.parse_args()

    send_generate_request(args.text)
