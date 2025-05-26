import os
import time
from pathlib import Path

import litserve as ls
from litserve import Request, Response

from detoxy.ml.config import CONFIG, SERVER_CONFIG
from detoxy.ml.module import ToxicClassifier


class SimpleLitAPI(ls.LitAPI):
    def setup(
        self,
        device: str,
        ckpt_path: str | Path = SERVER_CONFIG.finetuned,
        precision: str = SERVER_CONFIG.precision,
    ) -> None:
        self.lit_module = ToxicClassifier.load_from_checkpoint(ckpt_path).to(device)
        self.lit_module.to(device).to(precision)
        self.lit_module.eval()

    async def decode_request(self, request: Request):
        return request["input"]

    async def predict(self, input: str):
        return self.lit_module.predict_step(input)

    async def encode_response(self, output: dict) -> Response:
        return {"output": output}


class FileLogger(ls.Logger):
    def process(self, key, value):
        log_filepath = os.path.join(CONFIG.log_dir, "server_log.txt")
        with open(log_filepath, "a+") as f:
            f.write(f"{key}: {value:.2f}\n")


class PredictionTimeLogger(ls.Callback):
    def on_before_predict(self, lit_api):
        t0 = time.perf_counter()
        self._start_time = t0

    def on_after_predict(self, lit_api):
        t1 = time.perf_counter()
        elapsed = t1 - self._start_time
        lit_api.log("prediction_time", elapsed)


if __name__ == "__main__":
    api = SimpleLitAPI(enable_async=True)
    callbacks = [PredictionTimeLogger()]
    loggers = [FileLogger()]

    server = ls.LitServer(
        api,
        callbacks=callbacks,
        loggers=loggers,
        accelerator=SERVER_CONFIG.accelerator,
        devices=SERVER_CONFIG.devices,
        timeout=SERVER_CONFIG.timeout,
        track_requests=SERVER_CONFIG.track_requests,
    )
    server.run(port=8000, generate_client_file=SERVER_CONFIG.generate_client_file)
