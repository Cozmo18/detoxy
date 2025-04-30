from fastapi import Request, Response
from litserve import LitAPI, LitServer

from toxy_bot.ml.config import MODULE_CONFIG
from toxy_bot.ml.module import SequenceClassificationModule


class SimpleLitAPI(LitAPI):
    def setup(self, device: str):
        self.model = SequenceClassificationModule.load_from_checkpoint(
            MODULE_CONFIG.finetuned
        )
        self.model.to(device)
        self.model.eval()

    def decode_request(self, request: Request):
        return request["text"]

    def predict(self, text: str):
        return self.model.predict_step(text)

    def encode_response(self, output) -> Response:
        return {"output": output}


if __name__ == "__main__":
    server = LitServer(
        SimpleLitAPI(), accelerator="auto", devices=1, track_requests=True
    )
    server.run(port=8000)
