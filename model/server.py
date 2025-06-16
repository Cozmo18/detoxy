import os
from pathlib import Path

import litserve as ls
import torch
from litserve import Request, Response

from config import CONFIG, SERVER_CONFIG
from module import ToxicClassifier


class SimpleLitAPI(ls.LitAPI):
    def setup(
        self,
        device: str,
        checkpoint: str = SERVER_CONFIG.finetuned,
        ckpt_dir: str | Path = CONFIG.ckpt_dir,
    ) -> None:
        self.precision = torch.bfloat16
        self.lit_module = ToxicClassifier.load_from_checkpoint(os.path.join(ckpt_dir, checkpoint)).to(device)
        self.lit_module.to(device).to(self.precision)
        self.lit_module.eval()

        self.labels = self.lit_module.labels

    async def decode_request(self, request: Request):
        return request["input"]

    async def predict(self, input: str) -> torch.Tensor:
        return self.lit_module.predict_step(input)

    async def encode_response(self, output: torch.Tensor) -> Response:
        return {label: prob.item() for label, prob in zip(self.labels, output)}


if __name__ == "__main__":
    api = SimpleLitAPI(enable_async=True)
    server = ls.LitServer(
        api,
        accelerator=SERVER_CONFIG.accelerator,
        devices=SERVER_CONFIG.devices,
        timeout=SERVER_CONFIG.timeout,
        track_requests=SERVER_CONFIG.track_requests,
    )
    server.run(port=8000, generate_client_file=SERVER_CONFIG.generate_client_file)
