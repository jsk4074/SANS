# sans/ascent_ckpt.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple, Any, Dict
import types
import torch
import torch.nn as nn
from sans.audio_utils import WaveToMel
from sans.pipeline import style_transfer

AUDIO_PROMPT_TOKEN = "__AUDIO_CTX__"

# ------------ basic shape helpers ------------
def _ensure_b1t(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 1:  return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 2:  return x.unsqueeze(1)
    if x.ndim == 3:  return x if x.size(1) == 1 else x.mean(dim=1, keepdim=True)
    raise RuntimeError(f"wave should be [T]/[B,T]/[B,1,T], got {tuple(x.shape)}")

def _ensure_mel_b1ft(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float32)
    t = x
    if t.ndim == 4:
        if t.size(1) != 1: t = t.mean(dim=1, keepdim=True)
        return t.contiguous()
    if t.ndim == 3:
        B, A, B2 = t.shape
        return (t.unsqueeze(1) if A <= B2 else t.transpose(1,2).unsqueeze(1)).contiguous()
    if t.ndim == 2:
        F, T = t.shape
        return (t.unsqueeze(0).unsqueeze(0) if F <= T else t.t().unsqueeze(0).unsqueeze(0)).contiguous()
    if t.ndim == 1:
        return t.unsqueeze(0).unsqueeze(0).contiguous()
    raise RuntimeError(f"Unsupported spectrogram: {tuple(t.shape)}")

# ------------ build a text-like context from audio ------------
def _text_ctx_shape(ldm) -> Tuple[int,int]:
    dummy = ldm.get_learned_conditioning([""])
    return int(dummy.shape[1]), int(dummy.shape[2])

def _make_audio_context(ldm, e: torch.Tensor) -> torch.Tensor:
    N, D = _text_ctx_shape(ldm)
    if e.ndim >= 3:
        mel = _ensure_mel_b1ft(e).squeeze(1)     # [B,F,T]
        vec = mel.mean(dim=1)                     # [B,T]
    elif e.ndim == 2:
        vec = e                                   # [B,D’] treat D’ as “time”
    else:
        vec = e.unsqueeze(0)                      # [1,D’]
    B, Tin = vec.shape
    dev = vec.device

    key = "_audio2ctx_proj"
    if not hasattr(ldm, key):
        setattr(ldm, key, nn.Linear(Tin, D, bias=True).to(dev))
        nn.init.xavier_uniform_(getattr(ldm, key).weight, gain=0.5)
        nn.init.zeros_(getattr(ldm, key).bias)
    proj: nn.Linear = getattr(ldm, key)
    if proj.in_features != Tin or proj.out_features != D:
        new_proj = nn.Linear(Tin, D, bias=True).to(dev)
        nn.init.xavier_uniform_(new_proj.weight, gain=0.5)
        nn.init.zeros_(new_proj.bias)
        setattr(ldm, key, new_proj)
        proj = new_proj

    ctx1 = proj(vec)                # [B,D]
    ctx  = ctx1.unsqueeze(1)        # [B,1,D]
    if N > 1:
        ctx = ctx.expand(B, N, D).contiguous()
    return ctx

# ------------ public API ------------
def encode_audio_cond(ldm, wave: torch.Tensor, sr: int = 16000,
                      hop: int = 80, n_mels: int = 128) -> torch.Tensor:
    wave = _ensure_b1t(wave)
    to_mel = WaveToMel(sr=sr, hop=hop, n_mels=n_mels).to(wave.device)
    return to_mel(wave)  # [B,1,F,T]

def generate_with_audio_cond(
    ldm,
    e: torch.Tensor,
    steps: int = 12,
    guidance_scale: float = 2.5,
    duration_s: Optional[float] = None,
    seed: int = 1234,
    ref_path: Optional[str] = None,
    sr: int = 16000,
    output_type: str = "mel",
):
    """
    FORCE the sampler to attend to `e` by:
      1) Building audio cross-attn context `ctx`.
      2) Monkey-patching BOTH: (a) get_learned_conditioning, (b) the UNet forward,
         to inject `ctx` via any of the common entry points:
         - cond dict positional arg (c['c_crossattn'])
         - kwarg 'context'
         - kwarg 'encoder_hidden_states'
    """
    device = e.device if isinstance(e, torch.Tensor) else torch.device("cpu")
    ctx = _make_audio_context(ldm, e.to(device))  # [B,N,D]

    # --- 1) Hook get_learned_conditioning so our token returns ctx ---
    glc_orig = ldm.get_learned_conditioning
    def _glc_override(prompts):
        B = len(prompts) if isinstance(prompts, (list,tuple)) else 1
        # Use audio ctx iff our token present; else fall back for unconditional branch
        use_audio = False
        if isinstance(prompts, (list,tuple)):
            use_audio = any(str(p) == AUDIO_PROMPT_TOKEN for p in prompts)
        else:
            use_audio = (str(prompts) == AUDIO_PROMPT_TOKEN)
        if use_audio:
            out = ctx if ctx.size(0) == B else ctx[:1].expand(B, ctx.size(1), ctx.size(2)).contiguous()
            return out
        return glc_orig(prompts)

    # --- 2) Hook UNet.forward to hard-override context/conditioning ---
    # Try to locate the UNet-ish module (DiffusionWrapper / UNetModel)
    unet = None
    for name in ("model", "diffusion_model"):
        if hasattr(ldm, name):
            cand = getattr(ldm, name)
            if hasattr(cand, "forward"):
                unet = cand
                break
    if unet is None and hasattr(ldm, "model") and hasattr(ldm.model, "diffusion_model"):
        unet = ldm.model.diffusion_model
    if unet is None:
        # last resort: search modules
        for m in ldm.modules():
            if hasattr(m, "forward") and m.__class__.__name__.lower().find("unet") >= 0:
                unet = m
                break
        if unet is None and hasattr(ldm, "model"):
            unet = ldm.model  # hope this is DiffusionWrapper

    # wrap
    orig_forward = unet.forward if (unet is not None) else None
    printed_once = {"done": False}
    def wrapped_forward(*args, **kwargs):
        # Try kwarg routes
        if "context" in kwargs:
            kwargs["context"] = ctx
        if "encoder_hidden_states" in kwargs:
            kwargs["encoder_hidden_states"] = ctx
        # Try positional cond dict (CompVis-style)
        if len(args) >= 3 and isinstance(args[2], dict):
            cdict: Dict[str, Any] = dict(args[2])  # shallow copy
            if "c_crossattn" in cdict:
                cdict["c_crossattn"] = [ctx]
            args = (args[0], args[1], cdict, *args[3:])
        if not printed_once["done"]:
            keys = list(kwargs.keys())
            print(f"[inject] UNet.forward wrapped: args={len(args)} kwargs={keys}", flush=True)
            printed_once["done"] = True
        return orig_forward(*args, **kwargs)

    # install hooks
    ldm.get_learned_conditioning = _glc_override
    if orig_forward is not None:
        unet.forward = wrapped_forward

    try:
        # call your pipeline with our token so GLC is definitely hit
        out = style_transfer(
            ldm,
            AUDIO_PROMPT_TOKEN,
            original_audio_file_path=ref_path,
            transfer_strength=0.0,
            duration=duration_s if duration_s is not None else 5.0,
            guidance_scale=guidance_scale,
            ddim_steps=steps,
            output_type=output_type,
            # sr=sr,
        )
    finally:
        # restore hooks
        ldm.get_learned_conditioning = glc_orig
        if orig_forward is not None:
            unet.forward = orig_forward

    return out
