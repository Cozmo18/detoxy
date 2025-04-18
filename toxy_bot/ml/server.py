from litserve import LitAPI, LitServer
from fastapi import Request, Response

from toxy_bot.ml.config import CONFIG, MODULE_CONFIG
from toxy_bot.ml.module import SequenceClassificationModule

class SimpleLitAPI(LitAPI):
    def setup(self, device):
        # Load and move the model to the correct device
        self.lit_module = SequenceClassificationModule.load_from_checkpoint(MODULE_CONFIG.finetuned).to(device)
        # Keep track of the devices for moving data accordingly
        self.device = device
        
    def decode_request(self, request):
        return request["input"]
    
    def predict(self, sequence):
        return self.lit_module.predict_step(sequence)
    
    def encode_response(self, output) -> Response:
        return {"output": output}
    
    
if __name__ == "__main__":
    api = SimpleLitAPI()
    server = LitServer(api, accelerator="cuda", devices=1, timeout=30)
    server.run(port=8000)
    
    
        
    
        
        