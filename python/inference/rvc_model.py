# import monkey patch above all else
import logging
from threading import Lock
from typing import List, Optional

import numpy as np
import torch

from inference.algorithm.synthesizers import Synthesizer
from inference.api_models import CreateSongOptions
from inference.config import config
from inference.hubert import hubert_model
from inference.utils import load_audio
from inference.vc_infer_pipeline import VC

RVC_LOCK = Lock()

logger = logging.getLogger(__name__)


class RVCModel:
    def __init__(
        self,
        name: str,
        pth_files: List[str],
        index_files: List[str],
    ):
        self.name = name
        self.pth_files = pth_files
        self.index_files = index_files
        self.cpt = None

        if len(pth_files) == 0:
            raise RuntimeError(f"No .pth files found for {name}")
        logger.info(f"Available .pth files for {name}: {', '.join(pth_files)}")
        logger.info(f"Available .index files for {name}: {', '.join(index_files)}")
        for pth_file in self.pth_files:
            logger.info(f"Loading model file: {pth_file}")
            loaded_model = torch.load(pth_file, map_location=config.device)
            if loaded_model and "config" in loaded_model:
                logger.info(f"Using model file: {pth_file}")
                self.cpt = loaded_model
                break
        if self.cpt is None:
            raise RuntimeError(f"No valid .pth files found for {name}")

        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk

        if_f0 = self.cpt.get("f0", 1)
        logger.info(f"if_f0: {if_f0}")

        self.version = self.cpt.get("version", "v1")
        logger.info(f"Using version {self.version}")

        (
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
        ) = self.cpt["config"]

        logger.info(f"Model params")
        logger.info(f"spec_channels: {spec_channels}")
        logger.info(f"segment_size: {segment_size}")
        logger.info(f"inter_channels: {inter_channels}")
        logger.info(f"hidden_channels: {hidden_channels}")
        logger.info(f"filter_channels: {filter_channels}")
        logger.info(f"n_heads: {n_heads}")
        logger.info(f"n_layers: {n_layers}")
        logger.info(f"kernel_size: {kernel_size}")
        logger.info(f"p_dropout: {p_dropout}")
        logger.info(f"resblock: {resblock}")
        logger.info(f"resblock_kernel_sizes: {resblock_kernel_sizes}")
        logger.info(f"resblock_dilation_sizes: {resblock_dilation_sizes}")
        logger.info(f"upsample_rates: {upsample_rates}")
        logger.info(f"upsample_initial_channel: {upsample_initial_channel}")
        logger.info(f"upsample_kernel_sizes: {upsample_kernel_sizes}")
        logger.info(f"spk_embed_dim: {spk_embed_dim}")
        logger.info(f"gin_channels: {gin_channels}")
        logger.info(f"sr: {sr}")

        text_enc_hidden_dim = 768 if self.version == "v2" else 256
        vocoder = self.cpt.get("vocoder", "HiFi-GAN")
        use_f0 = True if if_f0 == 1 else False

        net_g = Synthesizer(
            *self.cpt["config"],
            use_f0=use_f0,
            text_enc_hidden_dim=text_enc_hidden_dim,
            vocoder=vocoder,
        )

        del net_g.enc_q
        net_g.load_state_dict(self.cpt["weight"], strict=False)

        net_g.eval().to(config.device)

        # Initialize the voice conversion and set n_spk
        self.vc = VC(
            self.tgt_sr, config, weights_path=None
        )  # weights_path will be set during inference
        self.n_spk = self.cpt["config"][-3]
        self.net_g = net_g.float()

    def clearMemory(self):
        del self.cpt
        del self.tgt_sr
        del self.vc
        del self.net_g
        del self.n_spk

    def run_inference(
        self,
        input_audio_path,
        weights_path,
        status_report,
        options: Optional[CreateSongOptions] = None,
    ):
        if input_audio_path is None:
            raise RuntimeError("No input audio path provided")

        status_report = status_report or (lambda x: None)
        file_index = self.index_files[0] if len(self.index_files) > 0 else ""
        options = options or CreateSongOptions()

        f0_up_key = options.pitch if options.pitch is not None else 0
        f0_method = options.f0Method if options.f0Method is not None else "rmvpe"
        index_rate = options.indexRatio if options.indexRatio is not None else 0.75
        protect = (
            options.consonantProtection
            if options.consonantProtection is not None
            else 0.35
        )
        volume_envelope = (
            options.volumeEnvelope if options.volumeEnvelope is not None else 1.0
        )
        sid = 0

        status_report("Loading audio...")
        audio = load_audio(input_audio_path, 16000)

        status_report("Loading hubert model...")
        hubert_model.load_model(weights_path)
        logger.info("Loaded hubert model")

        # Set weights_path for F0 models that need it
        self.vc.weights_path = weights_path

        if_f0 = self.cpt.get("f0", 1)
        pitch_guidance = True if if_f0 == 1 else False

        status_report("Performing inference...")

        audio_opt = self.vc.pipeline(
            model=hubert_model.hubert_model,
            net_g=self.net_g,
            sid=sid,
            audio=audio,
            pitch=f0_up_key,
            f0_method=f0_method,
            file_index=file_index,
            index_rate=index_rate,
            pitch_guidance=pitch_guidance,
            volume_envelope=volume_envelope,
            version=self.version,
            protect=protect,
            status_report=status_report,
        )

        return self.tgt_sr, audio_opt
