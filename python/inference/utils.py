import os

import librosa
import numpy as np
import soundfile as sf


def load_audio(file, sample_rate) -> np.ndarray:
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File not found: {file}")

        audio, sr = sf.read(file)

        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)

        if sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq"
            )

        return audio.flatten()
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file}: {e}")


def find_pth_and_index_files(directory):
    pth_files = []
    index_files = []

    for root, _, files in os.walk(directory):
        for filename in files:
            lower_filename = filename.lower()
            if lower_filename.endswith(".pth"):
                pth_files.append(os.path.join(root, filename))
            if lower_filename.endswith(".index"):
                index_files.append(os.path.join(root, filename))

    if len(pth_files) == 0:
        raise RuntimeError(f"No .pth files found in {directory}")
    return pth_files, index_files
