import gc
import os
import time
from functools import lru_cache
from threading import Lock
from typing import Callable

import torch
from audio_separator.separator import Separator

lock_dict = {}


class Stemmer:
    @staticmethod
    @lru_cache(maxsize=128)
    def _get_lock(source_audio_path: str, output_directory: str):
        key = (source_audio_path, output_directory)
        if key not in lock_dict:
            lock_dict[key] = Lock()
        return lock_dict[key]

    @staticmethod
    def separate_track(
        source_audio_path: str,
        output_directory: str,
        weights_dir: str,
        model_name: str,
        status_setter: Callable[[str], None] = None,
    ):
        if not os.path.exists(source_audio_path):
            raise Exception(f"Source audio path does not exist: {source_audio_path}")

        track_filename = os.path.basename(source_audio_path)
        track_name = os.path.splitext(track_filename)[0]

        def log_status(msg):
            if status_setter:
                status_setter(msg)
            print(msg)

        lock = Stemmer._get_lock(source_audio_path, output_directory)
        with lock:
            safe_model_name = "".join(x for x in model_name if x.isalnum())
            track_dir = os.path.join(output_directory, safe_model_name, track_name)
            os.makedirs(track_dir, exist_ok=True)

            expected_vocals = os.path.join(track_dir, "vocals.wav")
            expected_no_vocals = os.path.join(track_dir, "no_vocals.wav")

            if os.path.exists(expected_vocals) and os.path.exists(expected_no_vocals):
                log_status("Found existing separated stems. Skipping separation.")
                return expected_vocals, expected_no_vocals

            log_status(f"Initializing audio-separator for {model_name}...")
            separator = Separator(
                model_file_dir=weights_dir,
                output_dir=track_dir,
                output_format="WAV",
                use_autocast=True,
            )

            log_status(f"Loading model {model_name}...")
            separator.load_model(model_filename=model_name)

            custom_names = {
                "Vocals": "vocals",
                "Instrumental": "no_vocals",
                "Other": "no_vocals",
                "No Echo": "vocals",
                "Echo": "no_vocals",
                "No Reverb": "vocals",
                "Reverb": "no_vocals",
            }

            log_status("Separating track... (this may take a while)")
            start_time = time.time()

            output_filenames = separator.separate(
                source_audio_path, custom_output_names=custom_names
            )

            vocals_file = None
            no_vocals_file = None

            for f in output_filenames:
                full_path = os.path.join(track_dir, f) if not os.path.isabs(f) else f
                if "vocals.wav" in f:
                    vocals_file = full_path
                elif "no_vocals.wav" in f:
                    no_vocals_file = full_path

            if not vocals_file and len(output_filenames) > 0:
                vocals_file = (
                    os.path.join(track_dir, output_filenames[0])
                    if not os.path.isabs(output_filenames[0])
                    else output_filenames[0]
                )
            if not no_vocals_file and len(output_filenames) > 1:
                no_vocals_file = (
                    os.path.join(track_dir, output_filenames[1])
                    if not os.path.isabs(output_filenames[1])
                    else output_filenames[1]
                )

            elapsed_time = time.time() - start_time
            log_status(f"Separation complete. Elapsed time: {elapsed_time:.2f}s")

            del separator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return vocals_file, no_vocals_file
