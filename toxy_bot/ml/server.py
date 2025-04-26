import torch
from fastapi import Request, Response
from litserve import LitAPI, LitServer

from toxy_bot.ml.config import DATAMODULE_CONFIG, MODULE_CONFIG
from toxy_bot.ml.module import SequenceClassificationModule



class SimpleLitAPI(LitAPI):
    def setup(self, device):
        """
        Load the tokenizer and model, and move the model to the specified device.
        """
        # Load finetuned model 
        self.model = SequenceClassificationModule.load_from_checkpoint(MODULE_CONFIG.finetuned)
        
        # Move model to the device (e.g., CPU, GPU)
        self.model.to(device)
        
        # Set the model in evaluation mode
        self.model.eval()
        
   
    def decode_request(self, request: Request):
        """
        Preprocess the request data (tokenize)
        """
        return request["text"]
    
    def predict(self, text: str):
        return self.model.predict_step(text)

    def encode_response(self, output: torch.Tensor, threshold: float = 0.75, labels: list[str] = DATAMODULE_CONFIG.labels) -> Response:
        response = {}
        for idx, prob in enumerate(output.flatten()):
            if prob > threshold:
                label = labels[idx]
                response[label] = round(prob.item(), 4)

        return response



if __name__ == "__main__":
    api = SimpleLitAPI()
    server = LitServer(api, accelerator="auto", track_requests=True)
    server.run(port=8000)
