from pathlib import Path

import litserve as ls
import torch
from litserve import Request, Response

from detoxy.ml.config import MODULE_CONFIG
from detoxy.ml.module import ToxicClassifier


class SimpleLitAPI(ls.LitAPI):
    def setup(
        self, device: str, ckpt_path: str | Path = MODULE_CONFIG.finetuned
    ) -> None:
        # Load and move model to the correct device
        self.model = ToxicClassifier.load_from_checkpoint(ckpt_path)
        self.model.to(device)
        self.model.eval()

        # Keep track of the device for moving data accordingly
        self.device = device

    async def decode_request(self, request: Request) -> dict:
        return request["input"]

    async def predict(self, input: str) -> dict:
        return self.model.predict_step(input)

    async def encode_response(self, probabilities: torch.Tensor) -> Response:
        labels = self.model.labels
        probabilities = probabilities.flatten()

        return {label: prob.item() for label, prob in zip(labels, probabilities)}


if __name__ == "__main__":
    api = SimpleLitAPI(enable_async=True)
    server = ls.LitServer(api, accelerator="auto", track_requests=True)
    server.run(port=8000)
