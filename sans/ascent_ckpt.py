# sans/ascent_ckpt.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional

import torch

from sans.pipeline import style_transfer
from sans.audio_utils import WaveToMel


def _require(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def encode_audio_cond(ldm, wave: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """
    Compute the SAME audio-conditioning embedding used by your style_transfer path.
    Requires: ldm.audio_cond_encoder is bound and callable.
    Always passes mel [B,1,F,T] to the encoder.
    """
    _require(isinstance(wave, torch.Tensor), "wave must be a torch.Tensor")
    _require(wave.ndim in (2, 3), "wave shape must be [B,T] or [B,1,T]")
    _require(hasattr(ldm, "audio_cond_encoder") and callable(getattr(ldm, "audio_cond_encoder")),
             "ldm.audio_cond_encoder must exist and be callable")

    to_mel = WaveToMel(sr=int(sr))
    mel = to_mel(wave)  # [B,1,F,T]
    cond = ldm.audio_cond_encoder(mel)
    _require(isinstance(cond, torch.Tensor), "audio_cond_encoder must return a Tensor")
    return cond


def generate_with_audio_cond(
    ldm,
    cond_emb: torch.Tensor,
    *,
    steps: int = 12,
    guidance_scale: float = 2.5,
    duration_s: float = 5.0,
    seed: int = 1234,
    ref_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Generate audio using your sampler while *forcing* cond_emb as the
    audio-conditioning embedding (no re-encoding). We patch ldm.audio_cond_encoder.
    """
    _require(isinstance(cond_emb, torch.Tensor), "cond_emb must be a torch.Tensor")
    _require(hasattr(ldm, "audio_cond_encoder") and callable(getattr(ldm, "audio_cond_encoder")),
             "ldm.audio_cond_encoder must exist and be callable")

    enc_old = ldm.audio_cond_encoder
    def _return_cond_emb(*args, **kwargs):
        return cond_emb

    ldm.audio_cond_encoder = _return_cond_emb
    wav = style_transfer(
        ldm,
        "",  # no text
        original_audio_file_path=(ref_path if ref_path is not None else ""),
        transfer_strength=0.0,
        duration=duration_s,
        output_type="waveform",
        ddim_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    ldm.audio_cond_encoder = enc_old

    if not isinstance(wav, torch.Tensor):
        wav = torch.tensor(wav, dtype=torch.float32, device=cond_emb.device)

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.device != cond_emb.device:
        wav = wav.to(cond_emb.device)

    return wav
