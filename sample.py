#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa

from sans import build_model
from sans.audio_utils import WaveToMel
from sans.objectives import band_energy_objective
from sans.ascent_ckpt import encode_audio_cond, generate_with_audio_cond


# ---------------- audio normalization ----------------

def rms_dbfs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: [B, T] waveform in [-1,1]
    returns per-item RMS dBFS [B, 1]
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return 20.0 * torch.log10(rms.clamp_min(eps))

def normalize_rms(x: torch.Tensor, target_dbfs: float = -20.0, peak_clip: float = 0.999) -> torch.Tensor:
    """
    Scale each item to target RMS in dBFS. Clips to +/-peak_clip to avoid overflow.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    cur = rms_dbfs(x)                      # [B,1]
    gain_db = target_dbfs - cur            # [B,1]
    gain = (10.0 ** (gain_db / 20.0))      # [B,1]
    y = x * gain
    return y.clamp_(-peak_clip, peak_clip)

# ---------------- plotting: X=mel bins, Y=time ----------------

def _to_numpy_2d_ft(x: torch.Tensor | np.ndarray, n_mels_hint: int = 128) -> np.ndarray:
    """
    Normalize to a 2D array [F, T]. Uses n_mels_hint to decide which axis is F.
    Accepts shapes like [B,C,F,T], [B,F,T], [F,T], [T,F], etc.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.array(x)

    if a.ndim == 4:   # [B,C,F,T] or [B,C,T,F]
        a = a[0, 0]
    elif a.ndim == 3: # [B,F,T] or [B,T,F]
        a = a[0]
    elif a.ndim < 2:
        # last resort
        a = a.reshape(1, -1)

    if a.ndim != 2:
        a = a.reshape(a.shape[-2], a.shape[-1])  # take last two dims

    # Decide which axis is F by closeness to n_mels_hint
    F0, F1 = a.shape[0], a.shape[1]
    if abs(F0 - n_mels_hint) <= abs(F1 - n_mels_hint):
        ft = a  # [F,T] already
    else:
        ft = a.T  # [T,F] -> [F,T]
    return ft

def save_mel_image_swapped(
    mel_like: torch.Tensor | np.ndarray,
    path: str,
    *,
    sr: int = 16000,
    hop_length: int = 160,
    n_mels_hint: int = 128,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
):
    """
    Plot with **X = mel bins**, **Y = time (s)**.
    """
    try:
        img_ft = _to_numpy_2d_ft(mel_like, n_mels_hint=n_mels_hint)  # [F,T]
        img_tf = img_ft.T                                           # [T,F] for swapped axes

        if vmin is None or vmax is None:
            vmin = float(np.percentile(img_tf, 2.0))
            vmax = float(np.percentile(img_tf, 98.0))

        T, F = img_tf.shape
        t_max = (T - 1) * (hop_length / sr)

        plt.figure(figsize=(10, 4))
        extent = [0.0, float(F - 1), 0.0, t_max]  # X: mel bins, Y: seconds
        plt.imshow(img_tf, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(); cbar.set_label("log-mel")
        plt.xlabel("Mel bins"); plt.ylabel("Time (s)")
        if title: plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    except Exception as e:
        print(f"[save_mel_image_swapped] non-fatal error: {e}", file=sys.stderr)
    finally:
        plt.close()

# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser(description="CKPT-only ascent with RMS-normalized comparison (no HF)")
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
    parser.add_argument("--norm_dbfs", type=float, default=-20.0, help="RMS target (dBFS) for both plots")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}+cu{torch.version.cuda} | CUDA available: {torch.cuda.is_available()} | device: {device}")

    print(f"Loading AudioLDM CKPT: {args.ckpt}")
    ldm = build_model(args.ckpt)

    # Load reference audio
    print(f"Loading reference audio: {args.ref}")
    try:
        ref_wav, _ = librosa.load(args.ref, sr=args.sr)
    except Exception as e:
        print(f"[sample] Could not load {args.ref}: {e}. Using 440 Hz tone.", file=sys.stderr)
        t = np.linspace(0, args.duration, int(args.sr * args.duration), endpoint=False)
        ref_wav = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    ref_wave = torch.tensor(ref_wav, dtype=torch.float32, device=device).unsqueeze(0)

    # Clamp synth duration to reference length
    ref_len_s = max(0.25, len(ref_wav) / float(args.sr))
    eff_duration = min(args.duration, ref_len_s)

    print("Encoding audio condition (robust)...")
    cond0 = encode_audio_cond(ldm, ref_wave, sr=args.sr)
    if not isinstance(cond0, torch.Tensor):
        cond0 = torch.zeros((1, 768), dtype=torch.float32, device=device)

    # Objective
    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels)
    obj = band_energy_objective(to_mel, band=(args.band_lo, args.band_hi))

    # Ascent
    e = torch.nn.Parameter(cond0.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([e], lr=args.lr)

    last_out = None
    for t in range(args.ascent_steps):
        opt.zero_grad(set_to_none=True)

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

        with torch.no_grad():
            d = e - cond0
            n = d.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            over = (n > args.tau).float()
            e.data = (cond0 + d * (args.tau / n)) * over + e.data * (1 - over)

        if (t + 1) % 5 == 0:
            print(f"[{t+1}/{args.ascent_steps}] score={float(score):.6f}  loss={float(loss):.6f}")

    # ----- Build mel for FAIR comparison (RMS-normalized & shared scale) -----

    # Normalize reference waveform for plotting
    ref_wave_norm = normalize_rms(ref_wave, target_dbfs=args.norm_dbfs)
    mel_ref = to_mel(ref_wave_norm)                # [B,1,F,T] (log-mel)

    # Ascent output: if waveform, RMS-normalize too; if mel-like, just use it
    if isinstance(last_out, torch.Tensor) and last_out.ndim == 2 and last_out.size(1) > 4 * to_mel.n_fft:
        asc_wave_norm = normalize_rms(last_out, target_dbfs=args.norm_dbfs)
        mel_asc = to_mel(asc_wave_norm)
    else:
        mel_asc = last_out if isinstance(last_out, torch.Tensor) else torch.tensor(last_out, dtype=torch.float32)

    # Shared color scale (percentiles over both)
    img_ref = _to_numpy_2d_ft(mel_ref, n_mels_hint=args.n_mels)
    img_asc = _to_numpy_2d_ft(mel_asc, n_mels_hint=args.n_mels)
    both = np.concatenate([img_ref.flatten(), img_asc.flatten()])
    shared_vmin = float(np.percentile(both, 2.0))
    shared_vmax = float(np.percentile(both, 98.0))

    # Save figures (X = mel bins, Y = time)
    save_mel_image_swapped(mel_ref, "original_norm.png",
                           sr=args.sr, hop_length=args.hop, n_mels_hint=args.n_mels,
                           vmin=shared_vmin, vmax=shared_vmax, title="Original (normalized)")
    save_mel_image_swapped(mel_asc, "ascent_norm.png",
                           sr=args.sr, hop_length=args.hop, n_mels_hint=args.n_mels,
                           vmin=shared_vmin, vmax=shared_vmax, title="Ascent (normalized)")

    print("Done. Files written: original_norm.png, ascent_norm.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
