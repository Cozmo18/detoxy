from pathlib import Path
import time
import os

import litserve as ls
from litserve import Request, Response

from detoxy.ml.config import SERVER_CONFIG, CONFIG
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
        t0 = time.time()
        output = self.lit_module.predict_step(input)
        t1 = time.time()
        self.log("model_time", t1 - t0)
        return output

    async def encode_response(self, output: dict) -> Response:
        return {"output": output}
    
    
class SimpleLogger(ls.Logger):
    def process(self, key, value):
        print(f"Recieved {key} with value {value}", flush=True)
        
class FileLogger(ls.Logger):
    def process(self, key, value):
        log_filepath = os.path.join(CONFIG.log_dir, "server_logs.txt")
        with open(log_filepath, "a+") as f:
            f.write(f"{key}: {value:.1f}\n")
        

if __name__ == "__main__":
    api = SimpleLitAPI(enable_async=True)
    loggers = [SimpleLogger(), FileLogger()]
    server = ls.LitServer(
        api, 
        loggers=loggers,
        accelerator=SERVER_CONFIG.accelerator,
        devices=SERVER_CONFIG.devices,
        timeout=SERVER_CONFIG.timeout,
        track_requests=SERVER_CONFIG.track_requests,
    )
    server.run(port=8000)
