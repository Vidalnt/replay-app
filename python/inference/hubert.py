import os
import urllib.request
from threading import Lock

from torch import nn
from transformers import HubertModel

from inference.config import config

HUBERT_LOCK = Lock()


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class HubertModel:
    def __init__(self):
        self.hubert_model = None

    def load_model(self, weights_path: str, embedder_model: str = "contentvec"):
        with HUBERT_LOCK:
            if self.hubert_model is not None:
                return

            model_path = os.path.join(weights_path, "embedders", embedder_model)
            os.makedirs(model_path, exist_ok=True)

            bin_file = os.path.join(model_path, "pytorch_model.bin")
            json_file = os.path.join(model_path, "config.json")

            online_embedders = {
                "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin",
                "spin": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/pytorch_model.bin",
                "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/pytorch_model.bin",
            }

            config_files = {
                "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/config.json",
                "spin": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin/config.json",
                "spin-v2": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/config.json",
            }

            if not os.path.exists(bin_file):
                url = online_embedders.get(embedder_model)
                if url:
                    urllib.request.urlretrieve(url, bin_file)

            if not os.path.exists(json_file):
                url = config_files.get(embedder_model)
                if url:
                    urllib.request.urlretrieve(url, json_file)

            self.hubert_model = HubertModelWithFinalProj.from_pretrained(model_path)
            self.hubert_model = self.hubert_model.to(config.device).float()
            self.hubert_model.eval()


hubert_model = HubertModel()
