#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa

# SANS imports (these match what you showed)
from sans import build_model
from sans.pipeline import style_transfer, build_diffusers, ascent_transfer
from sans.audio_utils import WaveToMel
from sans.objectives import band_energy_objective
# (If you later want to use your AE objective:)
# from sans.objectives import make_ae_recon_objective

# --------------------------- utils -------------------------------------------

def save_mel_image(
    mel: torch.Tensor,
    path: str = "mel.png",
    sr: int = 16000,
    hop_length: int = 160,
    vmin: float = None,
    vmax: float = None,
    title: str = None,
):
    """Save a mel/fbank spectrogram tensor to an image."""
    if isinstance(mel, (list, tuple)):
        mel = mel[0]
    if isinstance(mel, torch.Tensor):
        mel = mel.detach().cpu()
    img = mel.numpy()

    # reduce dimensions: accept [B,1,F,T], [B,F,T], [F,T]
    if img.ndim == 4:                            # [B, C, F, T] or [B, C, T, F]
        # assume [B,C,F,T], take first batch & channel
        img = img[0, 0]
    if img.ndim == 3:                            # [B,F,T] -> first batch
        img = img[0]

    # robust display scaling
    if vmin is None or vmax is None:
        vmin = float(np.percentile(img, 2.0))
        vmax = float(np.percentile(img, 98.0))

    n_frames = img.shape[1]
    times = np.arange(n_frames) * (hop_length / sr)

    plt.figure(figsize=(10, 3))
    plt.imshow(img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label("log-mel")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel bins")
    if title:
        plt.title(title)
    xticks = np.linspace(0, n_frames - 1, 6)
    plt.xticks(xticks, [f"{t:.2f}" for t in (xticks * hop_length / sr)])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------- main --------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SANS sample: style_transfer vs ascent synthesis")
    parser.add_argument("--ckpt", type=str,
                        default="/home/tori/.cache/audioldm/audioldm-s-full.ckpt",
                        help="Path to AudioLDM ckpt for style_transfer path")
    parser.add_argument("--ref", type=str, default="trumpet.wav",
                        help="Reference audio file (wav, 16k recommended)")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate to load ref audio")
    parser.add_argument("--duration", type=float, default=5.0, help="Synthesis duration (seconds)")
    parser.add_argument("--hop", type=int, default=80, help="Mel hop length (samples)")
    parser.add_argument("--n_mels", type=int, default=128, help="Mel bins")
    parser.add_argument("--ascent_steps", type=int, default=25, help="Outer ascent steps")
    parser.add_argument("--inner_ddim_steps", type=int, default=12, help="Inner DDIM steps")
    parser.add_argument("--guidance", type=float, default=2.5, help="Guidance scale")
    parser.add_argument("--tau", type=float, default=2.0, help="Projection radius in embedding space")
    parser.add_argument("--lr", type=float, default=1e-2, help="Embedding ascent learning rate")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for deterministic DDIM")
    parser.add_argument("--band_lo", type=int, default=64, help="Objective band: lower mel bin (inclusive)")
    parser.add_argument("--band_hi", type=int, default=127, help="Objective band: upper mel bin (inclusive)")
    args = parser.parse_args()

    # Optional: if you've had CUDA env weirdness, you can force CPU by uncommenting:
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Basic env prints
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}")

    # 1) Load ckpt model (for style_transfer path)
    print(f"Load AudioLDM ckpt: {args.ckpt}")
    ldm = build_model(args.ckpt)

    # 2) Diffusers pipeline for embedding-ascent
    # NOTE: your build_diffusers() chooses device internally; no device arg here.
    print("Build Diffusers AudioLDM pipeline (for ascent path)...")
    pipe = build_diffusers("cvssp/audioldm-s-full-v2")

    # 3) Objective: simple high-band energy on mel (no AE needed)
    # Keep the mel config consistent with your visualization (sr/hop/n_mels).
    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels)
    obj = band_energy_objective(to_mel, band=(args.band_lo, args.band_hi))

    # 4) Run ascent (audio is generated in the loop; we return mel for convenience)
    print("Run ascent synthesis...")
    mel_ascent = ascent_transfer(
        pipe, "",  # empty text; ascent happens in the embedding space
        objective_fn=obj,
        ascent_steps=args.ascent_steps,
        inner_ddim_steps=args.inner_ddim_steps,
        guidance_scale=args.guidance,
        duration=args.duration,
        tau=args.tau,
        lr=args.lr,
        seed=args.seed,
        output_type="mel",
        to_mel=to_mel,   # ensures same mel config as the objective/visualization
    )

    # 5) Classic style_transfer (audio-to-audio via your ckpt path)
    print("Run style_transfer...")
    mel_style = style_transfer(
        ldm,
        "",  # no text prompt
        original_audio_file_path=args.ref,
        transfer_strength=0.3,
        duration=args.duration,
        output_type="mel",
    )

    # 6) Original ref -> mel for comparison
    print("Load & convert original to mel...")
    ref_wav, sr = librosa.load(args.ref, sr=args.sr)
    mel_ref = librosa.feature.melspectrogram(
        y=ref_wav, sr=args.sr, hop_length=args.hop, n_mels=args.n_mels
    )
    mel_ref = torch.tensor(mel_ref)

    # 7) Print shapes
    print("Shapes:")
    try:
        print("  original mel:", tuple(mel_ref.size()))
    except Exception:
        print("  original mel:", np.array(mel_ref).shape)
    try:
        print("  style mel:", tuple(mel_style.size()))
    except Exception:
        print("  style mel:", np.array(mel_style).shape)
    try:
        print("  ascent mel:", tuple(mel_ascent.size()))
    except Exception:
        print("  ascent mel:", np.array(mel_ascent).shape)

    # 8) Save images
    print("Save figures...")
    save_mel_image(mel_ref,     "original.png", sr=args.sr, hop_length=args.hop, title="Original")
    save_mel_image(mel_style,   "src_style.png", sr=args.sr, hop_length=args.hop, title="Style transfer")
    save_mel_image(mel_ascent,  "src_ascent.png", sr=args.sr, hop_length=args.hop, title="Ascent synthesis")

    print("Done. Files: original.png, src_style.png, src_ascent.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
