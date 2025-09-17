# sans/ascent_ckpt.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Tuple, Optional

import os
import tempfile

import torch
import numpy as np
from scipy.io import wavfile

from sans.pipeline import style_transfer
from sans.audio_utils import WaveToMel


# --------------------------- assertions ---------------------------------------

def _require(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


# --------------------------- discovery ----------------------------------------

def _looks_like_audio_encoder(path: str, clsname: str) -> bool:
    p = path.lower()
    c = clsname.lower()
    # Heuristics: prefer modules with "audio" or "clap" in the path/class,
    # and something encoder-ish.
    score = 0
    if "audio" in p or "clap" in p or "audio" in c or "clap" in c:
        score += 2
    if "enc" in p or "enc" in c or "encoder" in p or "encoder" in c:
        score += 2
    if "proj" in p or "proj" in c:
        score += 1
    return score >= 3


def find_audio_encoder_module(ldm: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
    """
    Scan model modules; return (module, dotted_path) for the best audio encoder candidate.
    """
    _require(isinstance(ldm, torch.nn.Module), "ldm must be a torch.nn.Module")

    best = None
    best_path = ""
    for name, mod in ldm.named_modules():
        clsname = mod.__class__.__name__
        if _looks_like_audio_encoder(name, clsname):
            # Prefer deeper paths (more specific)
            if (best is None) or (len(name) > len(best_path)):
                best = mod
                best_path = name

    _require(best is not None,
             "No audio encoder module found. Please ensure your CKPT includes a CLAP/audio encoder "
             "module (e.g., a submodule with 'audio'/'clap' and 'enc').")

    # Must have a callable forward
    _require(hasattr(best, "forward") and callable(best.forward),
             f"Candidate '{best_path}' exists but has no callable forward(...)")

    return best, best_path


# --------------------------- wave helpers -------------------------------------

def _ensure_wave_1ch_from_path(path: str, sr: int, length_s: float) -> np.ndarray:
    """
    Create a minimal placeholder wav if path is empty; otherwise leave IO to style_transfer.
    This is only used when we must feed style_transfer a path; if you pass a non-empty
    valid file path, style_transfer will read it (content will be ignored by patched encoder).
    """
    if isinstance(path, str) and len(path) > 0:
        # style_transfer will read this file itself
        return np.zeros((int(sr * 0.1),), dtype=np.float32)
    # Generate a tiny silence clip as placeholder
    n = int(sr * max(0.1, min(length_s, 1.0)))
    return np.zeros((n,), dtype=np.float32)


# --------------------------- public API ---------------------------------------

def encode_audio_cond(ldm: torch.nn.Module, wave: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """
    Compute the SAME audio-conditioning embedding your pipeline uses for style transfer.

    Steps:
      1) find audio encoder module inside ldm (by named_modules heuristics)
      2) build mel [B,1,F,T]
      3) call encoder.forward(mel) -> cond (Tensor)
    """
    _require(isinstance(wave, torch.Tensor), "wave must be a torch.Tensor")
    _require(wave.ndim in (2, 3), "wave shape must be [B,T] or [B,1,T]")

    enc_mod, enc_path = find_audio_encoder_module(ldm)

    to_mel = WaveToMel(sr=int(sr))
    mel = to_mel(wave)  # [B,1,F,T]

    cond = enc_mod.forward(mel)
    _require(isinstance(cond, torch.Tensor),
             f"audio encoder at '{enc_path}' must return a torch.Tensor")
    return cond


def generate_with_audio_cond(
    ldm: torch.nn.Module,
    cond_emb: torch.Tensor,
    *,
    steps: int = 12,
    guidance_scale: float = 2.5,
    duration_s: float = 5.0,
    seed: int = 1234,
    ref_path: Optional[str] = None,
    sr: int = 16000,
) -> torch.Tensor:
    """
    Generate audio by reusing style_transfer but forcing 'cond_emb' as the
    audio conditioning embedding (no re-encoding).

    Implementation: find the audio encoder module, temporarily replace its
    .forward with a function that returns cond_emb, call style_transfer to
    synthesize waveform, then restore the original .forward.
    """
    _require(isinstance(cond_emb, torch.Tensor), "cond_emb must be a torch.Tensor")

    enc_mod, enc_path = find_audio_encoder_module(ldm)
    _require(hasattr(enc_mod, "forward") and callable(enc_mod.forward),
             f"encoder at '{enc_path}' has no callable forward(...)")

    old_forward = enc_mod.forward

    def _return_cond_emb(*args, **kwargs):
        return cond_emb

    # patch
    enc_mod.forward = _return_cond_emb  # type: ignore[method-assign]

    # style_transfer requires a file path; pass the real ref if you have it,
    # or we'll create a tiny tmp wav below.
    use_tmp = not (isinstance(ref_path, str) and len(ref_path) > 0)

    if use_tmp:
        tmpdir = tempfile.mkdtemp(prefix="sans_ckpt_")
        tmpwav = os.path.join(tmpdir, "ref.wav")
        wav_np = _ensure_wave_1ch_from_path("", int(sr), duration_s)
        wavfile.write(tmpwav, int(sr), wav_np)
        in_path = tmpwav
    else:
        in_path = ref_path

    wav = style_transfer(
        ldm,
        "",  # no text
        original_audio_file_path=in_path,
        transfer_strength=0.0,
        duration=duration_s,
        output_type="waveform",
        ddim_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    # restore
    enc_mod.forward = old_forward  # type: ignore[method-assign]

    # cleanup
    if use_tmp:
        os.remove(in_path)
        os.rmdir(os.path.dirname(in_path))

    # Ensure Tensor [B, T], float32, device aligned with cond_emb
    if not isinstance(wav, torch.Tensor):
        wav = torch.tensor(wav, dtype=torch.float32, device=cond_emb.device)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.dtype != torch.float32:
        wav = wav.float()
    if wav.device != cond_emb.device:
        wav = wav.to(cond_emb.device)
    return wav
