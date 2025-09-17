#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa

from sans import build_model
from sans.audio_utils import WaveToMel
from sans.objectives import band_energy_objective
from sans.ascent_ckpt import encode_audio_cond, generate_with_audio_cond


def save_mel_image(
    mel: torch.Tensor,
    path: str = "mel.png",
    sr: int = 16000,
    hop_length: int = 160,
    vmin: float = None,
    vmax: float = None,
    title: str = None,
):
    if isinstance(mel, (list, tuple)):
        mel = mel[0]
    if isinstance(mel, torch.Tensor):
        mel = mel.detach().cpu()
    img = mel.numpy()

    if img.ndim == 4:   # [B, C, F, T]
        img = img[0, 0]
    if img.ndim == 3:   # [B, F, T]
        img = img[0]

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
    xticks = np.linspace(0, n_frames - 1, 6)
    plt.xticks(xticks, [f"{t:.2f}" for t in (xticks * hop_length / sr)])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def bind_audio_encoder(ldm):
    """
    Make sure ldm.audio_cond_encoder exists and is callable.
    Choose ONE of the discovered locations and bind it.
    """
    has = []
    if hasattr(ldm, "audio_cond_encoder"):
        if callable(getattr(ldm, "audio_cond_encoder")):
            return  # already bound

    if hasattr(ldm, "cond_stage_model"):
        if hasattr(ldm.cond_stage_model, "encode_audio"):
            has.append("cond_stage_model.encode_audio")

    if hasattr(ldm, "cond_stage_models"):
        models = ldm.cond_stage_models
        is_dict = isinstance(models, dict)
        has_audio_key = (("audio" in models) if is_dict else (hasattr(models, "__contains__") and ("audio" in models)))
        if has_audio_key:
            enc = models["audio"]
            if hasattr(enc, "forward") and callable(enc.forward):
                has.append("cond_stage_models['audio']")

    if hasattr(ldm, "audio_encoder"):
        if callable(getattr(ldm, "audio_encoder")):
            has.append("audio_encoder")

    if len(has) == 0:
        raise AssertionError(
            "No audio encoder found. Please expose one of:\n"
            "  ldm.audio_cond_encoder(mel)\n"
            "  ldm.cond_stage_model.encode_audio(mel)\n"
            "  ldm.cond_stage_models['audio'](mel)\n"
            "  ldm.audio_encoder(mel)\n"
        )

    choice = has[0]

    if choice == "cond_stage_model.encode_audio":
        ldm.audio_cond_encoder = ldm.cond_stage_model.encode_audio
    elif choice == "cond_stage_models['audio']":
        ldm.audio_cond_encoder = ldm.cond_stage_models["audio"]   # module; its forward(mel) will be used
    elif choice == "audio_encoder":
        ldm.audio_cond_encoder = ldm.audio_encoder
    else:
        raise AssertionError("Unexpected binding choice: " + choice)


def main():
    parser = argparse.ArgumentParser(description="CKPT-only ascent synthesis (no HF)")
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

    # Bind a concrete audio encoder for conditioning
    bind_audio_encoder(ldm)

    print(f"Loading reference audio: {args.ref}")
    ref_wav, _ = librosa.load(args.ref, sr=args.sr)
    wave = torch.tensor(ref_wav, dtype=torch.float32, device=device).unsqueeze(0)

    print("Encoding audio condition...")
    with torch.no_grad():
        cond0 = encode_audio_cond(ldm, wave, sr=args.sr)

    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels)
    obj = band_energy_objective(to_mel, band=(args.band_lo, args.band_hi))

    # Projected gradient ascent on the conditioning embedding
    e = torch.nn.Parameter(cond0.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([e], lr=args.lr)

    last_wave = None
    for t in range(args.ascent_steps):
        opt.zero_grad(set_to_none=True)

        # Reuse your sampler, bypassing re-encoding
        syn = generate_with_audio_cond(
            ldm,
            e,
            steps=args.inner_steps,
            guidance_scale=args.guidance,
            duration_s=args.duration,
            seed=args.seed,
            ref_path=args.ref,     # reuse the same file path; content is ignored by patched encoder
        )
        last_wave = syn

        score = obj(syn) if syn.ndim > 1 else obj(syn.unsqueeze(0))
        if score.ndim > 0:
            score = score.mean()

        reg = 1e-3 * (e - cond0).pow(2).mean()
        loss = -(score) + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_([e], 5.0)
        opt.step()

        # L2-ball projection around cond0
        with torch.no_grad():
            d = e - cond0
            n = d.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            over = (n > args.tau).float()
            e.data = (cond0 + d * (args.tau / n)) * over + e.data * (1 - over)

        if (t + 1) % 5 == 0:
            print(f"[{t+1}/{args.ascent_steps}] score={float(score):.6f}  loss={float(loss):.6f}")

    # Visualize
    mel_ref = librosa.feature.melspectrogram(
        y=ref_wav, sr=args.sr, hop_length=args.hop, n_mels=args.n_mels
    )
    mel_ref = torch.tensor(mel_ref)
    mel_ascent = to_mel(last_wave)

    save_mel_image(mel_ref,    "original.png",   sr=args.sr, hop_length=args.hop, title="Original (ref)")
    save_mel_image(mel_ascent, "src_ascent.png", sr=args.sr, hop_length=args.hop, title="Ascent (audio→audio)")

    print("Done. Files: original.png, src_ascent.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
