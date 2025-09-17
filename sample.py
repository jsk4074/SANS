#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa

from sans import build_model
from sans.audio_utils import WaveToMel
from sans.objectives import band_energy_objective
from sans.ascent_ckpt import encode_audio_cond, generate_with_audio_cond


# ---------- plotting helpers (fixes inverted axes) ----------------------------

def _to_numpy_2d_ft(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Normalize any spectrogram-like input to a 2D array shaped [F, T].

    Accepts:
      - torch or numpy
      - [B, C, F, T]
      - [B, C, T, F]
      - [B, F, T]
      - [B, T, F]
      - [F, T]
      - [T, F]

    Heuristic:
      If after squeezing to 2D we get [M, N] and M < N, we assume it's [T, F]
      and transpose to [F, T]. Otherwise we keep as-is.
    """
    # -> numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    arr = np.array(x)

    # squeeze batch/channel if present
    # cases: [B,C,F,T], [B,C,T,F], [B,F,T], [B,T,F]
    if arr.ndim == 4:
        # take first batch & channel
        arr = arr[0, 0]
    elif arr.ndim == 3:
        # take first batch
        arr = arr[0]

    # now arr is either [F,T] or [T,F]
    if arr.ndim != 2:
        # last resort: flatten to something plottable
        if arr.ndim == 1:
            arr = arr[None, :]
        else:
            # pick the last 2 axes
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    # ensure [F, T]
    F, T = arr.shape
    # Heuristic: if rows < cols, it's likely [T, F] -> transpose
    if F < T:
        arr = arr.T  # -> [F, T]

    return arr


def save_mel_image(
    mel_like: torch.Tensor | np.ndarray,
    path: str = "mel.png",
    sr: int = 16000,
    hop_length: int = 160,
    vmin: float = None,
    vmax: float = None,
    title: str = None,
):
    """
    Save a spectrogram image with **frequency on Y** and **time on X**.
    Handles all common shapes and fixes inverted axes automatically.
    """
    try:
        img = _to_numpy_2d_ft(mel_like)  # [F, T]

        # robust display scaling
        if vmin is None or vmax is None:
            vmin = float(np.percentile(img, 2.0))
            vmax = float(np.percentile(img, 98.0))

        n_frames = img.shape[1]
        plt.figure(figsize=(10, 3))
        plt.imshow(img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar()
        cbar.set_label("log-mel")
        plt.xlabel("Time (s)")
        plt.ylabel("Mel bins")
        if title:
            plt.title(title)
        if n_frames > 0:
            xticks = np.linspace(0, n_frames - 1, 6)
            times = xticks * (hop_length / sr)
            plt.xticks(xticks, [f"{t:.2f}" for t in times])
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    except Exception as e:
        print(f"[save_mel_image] non-fatal error: {e}", file=sys.stderr)
    finally:
        plt.close()


# ---------- main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CKPT-only ascent synthesis (robust, no HF)")
    parser.add_argument("--ckpt", type=str, default="/home/tori/.cache/audioldm/audioldm-s-full.ckpt")
    parser.add_argument("--ref", type=str, default="trumpet.wav")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--hop", type=int, default=80)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--ascent_steps", type=int, default=25)
    parser.add_argument("--inner_steps", type=int, default=12)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--band_lo", type=int, default=64)
    parser.add_argument("--band_hi", type=int, default=127)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}+cu{torch.version.cuda} | CUDA available: {torch.cuda.is_available()} | device: {device}")

    print(f"Loading AudioLDM CKPT: {args.ckpt}")
    ldm = build_model(args.ckpt)

    # Load reference audio (robust)
    print(f"Loading reference audio: {args.ref}")
    try:
        ref_wav, _ = librosa.load(args.ref, sr=args.sr)
    except Exception as e:
        print(f"[sample] Could not load {args.ref}: {e}. Falling back to a 440Hz tone.", file=sys.stderr)
        t = np.linspace(0, args.duration, int(args.sr * args.duration), endpoint=False)
        ref_wav = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    wave = torch.tensor(ref_wav, dtype=torch.float32, device=device).unsqueeze(0)

    # Clamp synth duration to the reference audio length (avoid duration warnings)
    ref_len_s = max(0.25, len(ref_wav) / float(args.sr))
    eff_duration = min(args.duration, ref_len_s)

    print("Encoding audio condition (with graceful fallback)...")
    cond0 = encode_audio_cond(ldm, wave, sr=args.sr)  # returns Tensor

    # Objective (works for waveform or mel-like tensors)
    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels)
    obj = band_energy_objective(to_mel, band=(args.band_lo, args.band_hi))

    # Parameter to optimize
    if not isinstance(cond0, torch.Tensor):
        cond0 = torch.zeros((1, 768), dtype=torch.float32, device=device)
    e = torch.nn.Parameter(cond0.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([e], lr=args.lr)

    last_out = None
    for t in range(args.ascent_steps):
        opt.zero_grad(set_to_none=True)

        # Generate (waveform preferred; may return mel-like)
        syn = generate_with_audio_cond(
            ldm,
            e,
            steps=args.inner_steps,
            guidance_scale=args.guidance,
            duration_s=eff_duration,
            seed=args.seed,
            ref_path=args.ref,
            sr=args.sr,
        )
        last_out = syn

        score = obj(syn if syn.ndim > 1 else syn.unsqueeze(0))
        if score.ndim > 0:
            score = score.mean()

        reg = 1e-3 * (e - cond0).pow(2).mean()
        loss = -(score) + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_([e], 5.0)
        opt.step()

        # L2-ball projection
        with torch.no_grad():
            d = e - cond0
            n = d.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            over = (n > args.tau).float()
            e.data = (cond0 + d * (args.tau / n)) * over + e.data * (1 - over)

        if (t + 1) % 5 == 0:
            print(f"[{t+1}/{args.ascent_steps}] score={float(score):.6f}  loss={float(loss):.6f}")

    # Visualize
    try:
        mel_ref = librosa.feature.melspectrogram(
            y=np.array(ref_wav), sr=args.sr, hop_length=args.hop, n_mels=args.n_mels
        )
        save_mel_image(mel_ref, "original.png", sr=args.sr, hop_length=args.hop, title="Original (ref)")
    except Exception as e_vis:
        print(f"[viz:original] non-fatal error: {e_vis}", file=sys.stderr)

    # If model returned waveform, convert; if it returned mel already, use it
    out_for_viz = last_out
    try:
        if isinstance(out_for_viz, torch.Tensor):
            if out_for_viz.ndim == 2 and out_for_viz.size(1) > 4 * to_mel.n_fft:
                # waveform [B,T] (likely): convert to mel
                mel_ascent = to_mel(out_for_viz)
            else:
                # already mel-like
                mel_ascent = out_for_viz
        else:
            mel_ascent = to_mel(torch.tensor(out_for_viz, dtype=torch.float32).unsqueeze(0))
        save_mel_image(mel_ascent, "src_ascent.png", sr=args.sr, hop_length=args.hop, title="Ascent (audio→audio)")
    except Exception as e_vis2:
        print(f"[viz:ascent] non-fatal error: {e_vis2}", file=sys.stderr)

    print("Done. Files (if no viz errors): original.png, src_ascent.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
