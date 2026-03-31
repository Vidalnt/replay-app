import os
from inference.api_models import StemmingModel
from inference.uvr.constants import DEMUCS_ARCH_TYPE, MDX_ARCH_TYPE, VR_ARCH_TYPE
from audio_separator.separator import Separator

stemming_models_list = []

try:
    separator = Separator(info_only=True, model_file_dir=os.path.join(os.getcwd(), "uvr"))
    all_models = separator.list_supported_model_files()
    
    ARCH_MAP = {
        "VR": VR_ARCH_TYPE,
        "MDX": MDX_ARCH_TYPE,
        "Demucs": DEMUCS_ARCH_TYPE
    }
    
    for arch_key, arch_type in ARCH_MAP.items():
        for model in all_models.get(arch_key, {}).values():
            if name := model.get("filename"):
                stemming_models_list.append(StemmingModel(name=name, files=[name], type=arch_type))
                
except Exception as e:
    print(f"Error loading stemming models: {e}")