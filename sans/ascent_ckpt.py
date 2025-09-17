# sans/ascent_ckpt.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Tuple, Callable
import sys
import os
import tempfile

import torch
import numpy as np
from scipy.io import wavfile

from sans.pipeline import style_transfer
from sans.audio_utils import WaveToMel


def _safe_log(msg: str):
    print(f"[ascent_ckpt] {msg}", file=sys.stderr)


def _looks_like_audio_encoder(path: str, clsname: str) -> bool:
    p = path.lower()
    c = clsname.lower()
    score = 0
    if ("audio" in p) or ("clap" in p) or ("audio" in c) or ("clap" in c):
        score += 2
    if ("enc" in p) or ("enc" in c) or ("encoder" in p) or ("encoder" in c):
        score += 2
    if ("proj" in p) or ("proj" in c):
        score += 1
    return score >= 3


def _find_audio_encoder_module(ldm: torch.nn.Module) -> Tuple[Optional[torch.nn.Module], str]:
    best = None
    best_path = ""
    try:
        for name, mod in ldm.named_modules():
            clsname = getattr(mod, "__class__", type(mod)).__name__
            if _looks_like_audio_encoder(name, clsname):
                if (best is None) or (len(name) > len(best_path)):
                    if hasattr(mod, "forward") and callable(mod.forward):
                        best = mod
                        best_path = name
    except Exception as e:
        _safe_log(f"module scan error: {e}")
    return best, best_path


def encode_audio_cond(ldm: torch.nn.Module, wave: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """
    Try hard to compute an audio-conditioning embedding.
    If we fail to discover/call an encoder, return a deterministic pooled mel as a fallback.
    """
    device = wave.device if isinstance(wave, torch.Tensor) else ("cuda" if torch.cuda.is_available() else "cpu")
    to_mel = WaveToMel(sr=int(sr))
    mel = None
    try:
        mel = to_mel(wave)  # [B,1,F,T]
    except Exception as e:
        _safe_log(f"mel transform failed: {e}. Falling back to zeros.")
        # fallback: simple zeros mel
        B = int(wave.shape[0]) if isinstance(wave, torch.Tensor) and wave.ndim >= 1 else 1
        mel = torch.zeros((B, 1, 128, 32), dtype=torch.float32, device=device)

    enc_mod, enc_path = _find_audio_encoder_module(ldm)
    if enc_mod is not None:
        try:
            cond = enc_mod.forward(mel)
            if not isinstance(cond, torch.Tensor):
                raise TypeError("encoder returned non-tensor")
            return cond
        except Exception as e:
            _safe_log(f"calling encoder '{enc_path}' failed: {e}. Using pooled-mel fallback.")

    # Fallback: pooled mel as a pseudo-embedding (stable shape)
    # [B,1,F,T] -> [B,F] by mean over T, then keep 2D
    try:
        m = mel
        if m.ndim == 4:
            m = m.mean(dim=-1)[:, 0]  # [B,F]
        elif m.ndim == 3:
            m = m.mean(dim=-1)        # [B,F]
        else:
            m = m.view(m.shape[0], -1)
        return m
    except Exception as e:
        _safe_log(f"pooled-mel fallback failed: {e}. Returning zeros [1,768].")
        return torch.zeros((1, 768), dtype=torch.float32, device=device)


def _make_returner(cond_emb: torch.Tensor) -> Callable:
    def _ret(*args, **kwargs):
        return cond_emb
    return _ret


def _ensure_ref_path_for_style(ref_path: Optional[str], sr: int, duration_s: float) -> Tuple[str, Optional[str]]:
    """
    style_transfer expects a file path. If you didn't pass one, create a short temp wav.
    Returns (path_to_use, tmpdir_or_None)
    """
    if isinstance(ref_path, str) and len(ref_path) > 0 and os.path.exists(ref_path):
        return ref_path, None
    tmpdir = tempfile.mkdtemp(prefix="sans_ckpt_")
    tmpwav = os.path.join(tmpdir, "ref.wav")
    n = int(sr * max(0.1, min(duration_s, 1.0)))
    wav_np = np.zeros((n,), dtype=np.float32)
    wavfile.write(tmpwav, int(sr), wav_np)
    return tmpwav, tmpdir


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
    Try to synthesize audio by patching the discovered encoder to return cond_emb.
    If patching fails or encoder not found, fall back to a normal style_transfer call.
    Always returns a Tensor waveform [B,T] on the same device as cond_emb.
    """
    device = cond_emb.device if isinstance(cond_emb, torch.Tensor) else ("cuda" if torch.cuda.is_available() else "cpu")

    enc_mod, enc_path = _find_audio_encoder_module(ldm)
    path_to_use, tmpdir = _ensure_ref_path_for_style(ref_path, int(sr), duration_s)

    wav = None
    patched = False
    try:
        if enc_mod is not None and hasattr(enc_mod, "forward") and callable(enc_mod.forward):
            old_forward = enc_mod.forward
            enc_mod.forward = _make_returner(cond_emb)  # patch
            patched = True

        wav = style_transfer(
            ldm,
            "",  # no text
            original_audio_file_path=path_to_use,
            transfer_strength=0.0,
            duration=duration_s,
            output_type="waveform",
            ddim_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        if patched:
            enc_mod.forward = old_forward  # restore
    except Exception as e:
        _safe_log(f"style_transfer with patched encoder failed: {e}. Falling back to unpatched run.")
        try:
            if patched:
                enc_mod.forward = old_forward  # restore if needed
            wav = style_transfer(
                ldm,
                "",  # no text
                original_audio_file_path=path_to_use,
                transfer_strength=0.0,
                duration=duration_s,
                output_type="waveform",
                ddim_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        except Exception as e2:
            _safe_log(f"style_transfer fallback failed: {e2}. Returning zeros.")
            wav = torch.zeros((1, int(sr * duration_s)), dtype=torch.float32, device=device)

    # cleanup tmp
    try:
        if tmpdir is not None:
            os.remove(path_to_use)
            os.rmdir(tmpdir)
    except Exception:
        pass

    # ensure Tensor [B,T] float32 on device
    if not isinstance(wav, torch.Tensor):
        try:
            wav = torch.tensor(wav, dtype=torch.float32, device=device)
        except Exception:
            wav = torch.zeros((1, int(sr * duration_s)), dtype=torch.float32, device=device)

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.dtype != torch.float32:
        wav = wav.float()
    if wav.device != device:
        wav = wav.to(device)

    return wav
