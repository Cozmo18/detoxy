import os
from pathlib import Path

import litserve as ls
import torch
from litserve import Request, Response

from model.config import AppConfig, Config
from model.module import SequenceClassificationModule


class SimpleLitAPI(ls.LitAPI):
    def setup(
        self,
        device: str,
        checkpoint: str = AppConfig.finetuned,
        ckpt_dir: str | Path = Config().ckpt_dir,
    ) -> None:
        self.precision = torch.bfloat16
        self.lit_module = SequenceClassificationModule.load_from_checkpoint(
            os.path.join(ckpt_dir, checkpoint)
        ).to(device)
        self.lit_module.to(device).to(self.precision)
        self.lit_module.eval()

        self.labels = self.lit_module.labels

    async def decode_request(self, request: Request):
        return request["input"]

    async def predict(self, input: str) -> torch.Tensor:
        return self.lit_module.predict_step(input)

    async def encode_response(self, output: torch.Tensor) -> Response:
        return {
            label: prob.item() for label, prob in zip(self.labels, output, strict=False)
        }


def main() -> None:
    api = SimpleLitAPI(enable_async=True)
    server = ls.LitServer(
        api,
        accelerator=AppConfig.accelerator,
        devices=AppConfig.devices,
        timeout=AppConfig.timeout,
        track_requests=AppConfig.track_requests,
    )
    server.run(port=8000, generate_client_file=AppConfig.generate_client_file)
