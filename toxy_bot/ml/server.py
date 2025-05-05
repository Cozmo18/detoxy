from fastapi import Request, Response
from litserve import LitAPI, LitServer
from pathlib import Path

from toxy_bot.ml.config import MODULE_CONFIG
from toxy_bot.ml.module import SequenceClassificationModule

import torch


class SimpleLitAPI(LitAPI):
    def setup(
        self, device: str, ckpt_path: str | Path = MODULE_CONFIG.finetuned
    ) -> None:
        # Load and move model to the correct device
        self.model = SequenceClassificationModule.load_from_checkpoint(ckpt_path)
        self.model.to(device)
        self.model.eval()

        # Keep track of the device for moving data accordingly
        self.device = device

    def decode_request(self, request: Request) -> dict:
        return request["input"]

    def predict(self, input: str) -> dict:
        return self.model.predict_step(input)

    def encode_response(self, probabilities: torch.Tensor) -> Response:
        labels = self.model.labels
        probabilities = probabilities.flatten()

        return {label: prob.item() for label, prob in zip(labels, probabilities)}


if __name__ == "__main__":
    server = LitServer(
        SimpleLitAPI(), accelerator="auto", devices=1, track_requests=True
    )
    server.run(port=8000)
