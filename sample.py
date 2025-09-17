#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, warnings, sys, numpy as np, matplotlib.pyplot as plt, torch, librosa
from sans import build_model
from sans.audio_utils import WaveToMel
from sans.objectives import band_energy_objective
from sans.ascent_ckpt import encode_audio_cond, generate_with_audio_cond

from tqdm import tqdm

# ---------------- audio normalization ----------------
def rms_dbfs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if x.ndim == 1: x = x.unsqueeze(0)
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return 20.0 * torch.log10(rms.clamp_min(eps))

def normalize_rms(x: torch.Tensor, target_dbfs: float = -20.0, peak_clip: float = 0.999) -> torch.Tensor:
    if x.ndim == 1: x = x.unsqueeze(0)
    gain = 10.0 ** ((target_dbfs - rms_dbfs(x)) / 20.0)
    y = x * gain
    return y.clamp_(-peak_clip, peak_clip)

# ---------------- shape utils: ALWAYS return [B,1,F,T] ----------------
def ensure_mel_b1ft(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts waveform [B,T] or mel-like [B,1,F,T]/[B,F,T]/[F,T]/[B,T,F].
    Returns mel [B,1,F,T]. If input is waveform, caller should run to_mel() first.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    t = x
    if t.ndim == 4:                    # [B,1,F,T] or [B,C,F,T]
        if t.size(1) != 1:
            t = t.mean(dim=1, keepdim=True)
        return t
    if t.ndim == 3:                    # [B,F,T] or [B,T,F]
        # decide if middle is F or T by which is smaller (F ~ 64..512 typical)
        B, A, B2 = t.shape
        if A <= B2:    # [B,F,T]
            return t.unsqueeze(1)
        else:          # [B,T,F] -> [B,1,F,T]
            return t.transpose(1, 2).unsqueeze(1)
    if t.ndim == 2:                    # [F,T] or [T,F]
        F, T = t.shape
        if F <= T:    # [F,T]
            return t.unsqueeze(0).unsqueeze(0)
        else:         # [T,F]
            return t.t().unsqueeze(0).unsqueeze(0)
    if t.ndim == 1:                    # [T] -> fake batch/ch
        return t.unsqueeze(0).unsqueeze(0)
    raise RuntimeError(f"Unsupported tensor shape for ensure_mel_b1ft: {tuple(t.shape)}")

# ---------------- plotting with explicit axes ----------------
def plot_mel_axes(
        mel_b1ft: torch.Tensor, 
        *, 
        path: str, 
        sr: int, 
        hop: int,
        axes: str = "time-x", 
        title: str = None, 
        vmin=None, 
        vmax=None
    ):
    """
    mel_b1ft: [B,1,F,T]
    axes: "time-x"  -> X=time(s),  Y=mel bins
          "time-y"  -> X=mel bins, Y=time(s)
    """
    m = mel_b1ft.detach().cpu().numpy()
    if m.ndim != 4 or m.shape[1] != 1:
        m = ensure_mel_b1ft(torch.tensor(m)).numpy()
    img = m[0, 0]  # [F,T]

    F, T = img.shape
    if vmin is None or vmax is None:
        vmin = float(np.percentile(img, 2.0)); vmax = float(np.percentile(img, 98.0))
    t_max = (T - 1) * (hop / sr)

    plt.figure(figsize=(10, 4))
    if axes == "time-x":
        # X: time(s), Y: mel bins
        extent = [0.0, t_max, 0.0, float(F - 1)]
        plt.imshow(img, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        plt.xlabel("Time (s)"); plt.ylabel("Mel bins")
    else:
        # axes == "time-y": X: mel bins, Y: time(s)
        extent = [0.0, float(F - 1), 0.0, t_max]
        plt.imshow(img.T, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
        plt.xlabel("Mel bins"); plt.ylabel("Time (s)")

    cbar = plt.colorbar(); cbar.set_label("log-mel")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="CKPT-only ascent with explicit axes + normalization")
    ap.add_argument("--ckpt", type=str, default="/home/tori/.cache/audioldm/audioldm-s-full.ckpt")
    ap.add_argument("--ref", type=str, default="trumpet.wav")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--hop", type=int, default=80)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--ascent_steps", type=int, default=10)
    ap.add_argument("--inner_steps", type=int, default=12)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--band_lo", type=int, default=64)
    ap.add_argument("--band_hi", type=int, default=127)
    ap.add_argument("--norm_dbfs", type=float, default=-20.0)
    ap.add_argument("--axes", type=str, default="time-x", choices=["time-x","time-y"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}+cu{torch.version.cuda} | CUDA={torch.cuda.is_available()} | device={device}")

    # Load model
    ldm = build_model(args.ckpt)

    # Load reference audio
    try:
        ref_wav, _ = librosa.load(args.ref, sr=args.sr)
    except Exception as e:
        print(f"[sample] load failed ({e}); using 440Hz tone", file=sys.stderr)
        t = np.linspace(0, args.duration, int(args.sr * args.duration), endpoint=False)
        ref_wav = 0.2 * np.sin(2*np.pi*440.0*t).astype(np.float32)
    ref_wave = torch.tensor(ref_wav, dtype=torch.float32, device=device).unsqueeze(0)

    # Duration clamp
    ref_len_s = max(0.25, len(ref_wav) / float(args.sr))
    eff_duration = min(args.duration, ref_len_s)

    # Encode conditioning (robust helper from your repo)
    with torch.no_grad():
        cond0 = encode_audio_cond(ldm, ref_wave, sr=args.sr)
    if not isinstance(cond0, torch.Tensor):
        cond0 = torch.zeros((1, 768), dtype=torch.float32, device=device)

    # Objective & mel transformer
    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels)
    obj = band_energy_objective(to_mel, band=(args.band_lo, args.band_hi))

    # Ascent
    e = torch.nn.Parameter(cond0.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([e], lr=args.lr)

    last_out = None
    for it in tqdm(range(args.ascent_steps)):
        opt.zero_grad(set_to_none=True)
        syn = generate_with_audio_cond(
            ldm, e, steps=args.inner_steps, guidance_scale=args.guidance,
            duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr
        )
        last_out = syn
        score = obj(syn if syn.ndim > 1 else syn.unsqueeze(0))
        if score.ndim: score = score.mean()
        reg = 1e-3 * (e - cond0).pow(2).mean()
        loss = -(score) + reg
        loss.backward(); torch.nn.utils.clip_grad_norm_([e], 5.0); opt.step()
        with torch.no_grad():
            d = e - cond0; n = d.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            e.data = cond0 + d * (args.tau / n).clamp(max=1.0)
        # if (it+1) % 5 == 0:
        #     print(f"[{it+1}/{args.ascent_steps}] score={float(score):.6f}  loss={float(loss):.6f}")

    # ----- Build normalized mels in canonical [B,1,F,T] -----
    ref_wave_norm = normalize_rms(ref_wave, target_dbfs=args.norm_dbfs)
    mel_ref = ensure_mel_b1ft(to_mel(ref_wave_norm.to("cpu")))        # [B,1,F,T]

    if isinstance(last_out, torch.Tensor) and last_out.ndim == 2 and last_out.size(1) > 4*to_mel.n_fft:
        # asc_wave_norm = last_out
        # asc_wave_norm = normalize_rms(last_out, target_dbfs=args.norm_dbfs)
        mel_asc = ensure_mel_b1ft(to_mel(last_out))
    else:
        mel_asc = ensure_mel_b1ft(last_out)

    # Shared color scale
    ref_img = mel_ref[0,0].detach().cpu().numpy()
    asc_img = mel_asc[0,0].detach().cpu().numpy()
    both = np.concatenate([ref_img.ravel(), asc_img.ravel()])
    vmin = float(np.percentile(both, 2.0)); vmax = float(np.percentile(both, 98.0))

    # Plots with explicit axes
    plot_mel_axes(
        mel_ref, 
        path="original_norm.png", 
        sr=args.sr, 
        hop=args.hop,
        axes=args.axes, 
        title="Original (normalized)", 
        vmin=vmin, 
        vmax=vmax
    )
    plot_mel_axes(
        mel_asc, 
        path="ascent_norm.png", 
        sr=args.sr, 
        hop=args.hop,
        axes=args.axes, 
        title="Ascent (normalized)",   
        vmin=vmin, 
        vmax=vmax
    )

    print(f"Done. Axes mode: {args.axes}. Files: original_norm.png, ascent_norm.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
