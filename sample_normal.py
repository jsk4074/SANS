#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, warnings, sys, numpy as np, matplotlib.pyplot as plt, torch, librosa

from sans import build_model
from sans.audio_utils import WaveToMel
from sans.ascent_ckpt import encode_audio_cond, generate_with_audio_cond


# ---------------- normalization (ONLY for original) ----------------
def rms_dbfs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return 20.0 * torch.log10(rms.clamp_min(eps))

def normalize_rms(x: torch.Tensor, target_dbfs: float = -20.0, peak_clip: float = 0.999) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    gain = 10.0 ** ((target_dbfs - rms_dbfs(x)) / 20.0)
    y = x * gain
    return y.clamp_(-peak_clip, peak_clip)

# ---------------- unify spectrogram to [B,1,F,T] ----------------
def ensure_mel_b1ft(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    t = x
    if t.ndim == 4:                      # [B,C,F,T]
        if t.size(1) != 1:
            t = t.mean(dim=1, keepdim=True)
        return t
    if t.ndim == 3:                      # [B,F,T] or [B,T,F]
        B, A, B2 = t.shape
        return t.unsqueeze(1) if A <= B2 else t.transpose(1, 2).unsqueeze(1)
    if t.ndim == 2:                      # [F,T] or [T,F]
        F, T = t.shape
        return t.unsqueeze(0).unsqueeze(0) if F <= T else t.t().unsqueeze(0).unsqueeze(0)
    if t.ndim == 1:                      # [T]
        return t.unsqueeze(0).unsqueeze(0)
    raise RuntimeError(f"Unsupported tensor shape for ensure_mel_b1ft: {tuple(t.shape)}")

# ---------------- use WaveToMel only for wave; pass-through for specs ------------
def to_mel_or_passthrough(x: torch.Tensor, to_mel: torch.nn.Module) -> torch.Tensor:
    """
    If x is a waveform ([T], [B,T], [B,1,T]) -> compute mel with to_mel.
    If x is already spectrogram-like -> just coerce to [B,1,F,T].
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # waveform cases
    if x.ndim == 1:                # [T]
        return ensure_mel_b1ft(to_mel(x.unsqueeze(0)))
    if x.ndim == 2 and x.shape[0] <= 4:  # [B,T] (small B)
        return ensure_mel_b1ft(to_mel(x))
    if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] > 8:  # [B,1,T]
        return ensure_mel_b1ft(to_mel(x.squeeze(1)))

    # otherwise assume spectrogram-like
    return ensure_mel_b1ft(x)

# ---------------- simple mel plot (time on X) -----------------------------------
def plot_mel_time_x(mel_b1ft: torch.Tensor, *, path: str, sr: int, hop: int, title: str = None, vmin=None, vmax=None):
    m = ensure_mel_b1ft(mel_b1ft).detach().cpu().numpy()
    img = m[0, 0]  # [F,T]
    F, T = img.shape
    t_x = (T - 1) * (hop / sr)

    if vmin is None or vmax is None:
        vmin = float(np.percentile(img, 2.0)); vmax = float(np.percentile(img, 98.0))

    extent = [0.0, t_x, 0.0, float(F - 1)]
    plt.figure(figsize=(10, 4))
    plt.imshow(img, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(); cbar.set_label("log-mel")
    plt.xlabel("Time (s)"); plt.ylabel("Mel bins")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ---------------- main ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Baseline sample: no FCAE, no ascent. One-shot generation.")
    ap.add_argument("--ckpt", type=str, default="/home/tori/.cache/audioldm/audioldm-s-full.ckpt")
    ap.add_argument("--ref", type=str, default="trumpet.wav")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=5.0, help="Requested seconds; may be clamped to ref length.")
    ap.add_argument("--hop", type=int, default=80)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--inner_steps", type=int, default=12)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--norm_dbfs", type=float, default=-20.0, help="RMS target for ORIGINAL ONLY")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}+cu{torch.version.cuda} | CUDA={torch.cuda.is_available()} | device={device}")

    # Model
    ldm = build_model(args.ckpt)

    # Reference audio & duration clamp
    ref_wav, _ = librosa.load(args.ref, sr=args.sr)
    ref_wave = torch.tensor(ref_wav, dtype=torch.float32, device=device).unsqueeze(0)
    ref_len_s = max(0.25, len(ref_wav) / float(args.sr))
    eff_duration = min(args.duration, ref_len_s)

    # Encode conditioning
    with torch.no_grad():
        cond = encode_audio_cond(ldm, ref_wave, sr=args.sr)
    print(f"[shapes] cond: {tuple(cond.shape)}")

    # One-shot generation with cond (no ascent, no FCAE)
    out = generate_with_audio_cond(
        ldm, cond, steps=args.inner_steps, guidance_scale=args.guidance,
        duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr
    )
    print(f"[shapes] generate output: {tuple(out.shape)}")

    # Build MELs
    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels).to(device)

    ref_wave_norm = normalize_rms(ref_wave, target_dbfs=args.norm_dbfs)
    mel_ref = ensure_mel_b1ft(to_mel(ref_wave_norm))
    mel_out = to_mel_or_passthrough(out, to_mel)

    # Per-image scaling to keep both readable
    vmin_ref, vmax_ref = np.percentile(mel_ref[0,0].detach().cpu().numpy(), [2, 98]).tolist()
    vmin_out, vmax_out = np.percentile(mel_out[0,0].detach().cpu().numpy(), [2, 98]).tolist()

    # Save images + tensors
    plot_mel_time_x(mel_ref, path="normal_original_norm.png", sr=args.sr, hop=args.hop, title="Original (RMS-normalized)", vmin=vmin_ref, vmax=vmax_ref)
    plot_mel_time_x(mel_out, path="normal_output.png", sr=args.sr, hop=args.hop, title="Model output", vmin=vmin_out, vmax=vmax_out)

    torch.save(mel_ref.cpu(), "normal_original_norm_mel.pt")
    torch.save(mel_out.cpu(), "normal_output_mel.pt")

    print("Wrote: normal_original_norm.png, normal_output.png, normal_original_norm_mel.pt, normal_output_mel.pt")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
