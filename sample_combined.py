#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, warnings, sys, numpy as np, matplotlib.pyplot as plt, torch, librosa
from tqdm import tqdm

from sans import build_model
from sans.audio_utils import WaveToMel
from sans.objectives import band_energy_objective
from sans.ascent_ckpt import encode_audio_cond, generate_with_audio_cond
from sans.fc_autoencoder import FcAutoEncoder  # optional

# ---------------- normalization (ONLY for original) ----------------
def rms_dbfs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if x.ndim == 1: x = x.unsqueeze(0)
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return 20.0 * torch.log10(rms.clamp_min(eps))

def normalize_rms(x: torch.Tensor, target_dbfs: float = -20.0, peak_clip: float = 0.999) -> torch.Tensor:
    if x.ndim == 1: x = x.unsqueeze(0)
    gain = 10.0 ** ((target_dbfs - rms_dbfs(x)) / 20.0)
    y = x * gain
    return y.clamp_(-peak_clip, peak_clip)

# ---------------- spectrogram coercion to [B,1,F,T] ----------------
def ensure_mel_b1ft(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    t = x
    if t.ndim == 4:  # [B,C,F,T]
        if t.size(1) != 1: t = t.mean(dim=1, keepdim=True)
        return t
    if t.ndim == 3:  # [B,F,T] or [B,T,F]
        B, A, B2 = t.shape
        return t.unsqueeze(1) if A <= B2 else t.transpose(1, 2).unsqueeze(1)
    if t.ndim == 2:  # [F,T] or [T,F]
        F, T = t.shape
        return t.unsqueeze(0).unsqueeze(0) if F <= T else t.t().unsqueeze(0).unsqueeze(0)
    if t.ndim == 1:  # [T]
        return t.unsqueeze(0).unsqueeze(0)
    raise RuntimeError(f"Unsupported tensor shape for ensure_mel_b1ft: {tuple(t.shape)}")

# ---------------- WaveToMel for waves; pass-through for specs -------------------
def to_mel_or_passthrough(x: torch.Tensor, to_mel: torch.nn.Module) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # waveform heuristics
    if x.ndim == 1:
        return ensure_mel_b1ft(to_mel(x.unsqueeze(0)))
    if x.ndim == 2 and x.shape[0] <= 4:         # [B,T]
        return ensure_mel_b1ft(to_mel(x))
    if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] > 8:  # [B,1,T]
        return ensure_mel_b1ft(to_mel(x.squeeze(1)))

    # otherwise already spectrogram-like
    return ensure_mel_b1ft(x)

# ---------------- differentiable dim adapters (repeat/slice) --------------------
def repeat_to_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Map last dim D -> target_dim by repeating then slicing (fully differentiable).
    """
    d = int(x.shape[-1])
    if d == target_dim:
        return x
    if d > target_dim:
        return x[..., :target_dim].contiguous()
    # d < target_dim: repeat
    reps = (target_dim + d - 1) // d
    x_rep = x.repeat_interleave(reps, dim=-1)  # [..., reps*d]
    return x_rep[..., :target_dim].contiguous()

def adapter_to_from(x: torch.Tensor, to_dim: int, back_dim: int) -> torch.Tensor:
    """
    Convenience: x (..., D) -> (..., to_dim) via repeat_to_dim, then back to (..., back_dim).
    Used to keep UNet-facing dim unchanged while letting FCAE operate at its native in_dim.
    """
    return repeat_to_dim(repeat_to_dim(x, to_dim), back_dim)

# ---------------- plotting (SWAPPED axes: X=mel, Y=time) -----------------------
def plot_mel_time_y(mel_b1ft: torch.Tensor, *, path: str, sr: int, hop: int, title: str = None, vmin=None, vmax=None):
    m = ensure_mel_b1ft(mel_b1ft).detach().cpu().numpy()
    img = m[0, 0]  # [F,T]
    F, T = img.shape
    data = img.T
    t_y = (T - 1) * (hop / sr)

    if vmin is None or vmax is None:
        vmin = float(np.percentile(data, 2.0)); vmax = float(np.percentile(data, 98.0))

    extent = [0.0, float(F - 1), 0.0, t_y]
    plt.figure(figsize=(6, 6))
    plt.imshow(data, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(); cbar.set_label("log-mel")
    plt.xlabel("Mel bins"); plt.ylabel("Time (s)")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ---------------- main ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Combined: baseline (normal) + ascent with optional FCAE (axes swapped)")
    ap.add_argument("--ckpt", type=str, default="/home/tori/.cache/audioldm/audioldm-s-full.ckpt")
    ap.add_argument("--ref", type=str, default="trumpet.wav")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--hop", type=int, default=80)
    ap.add_argument("--n_mels", type=int, default=128)

    # ascent params
    ap.add_argument("--ascent_steps", type=int, default=25)
    ap.add_argument("--inner_steps", type=int, default=12)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--band_lo", type=int, default=64)
    ap.add_argument("--band_hi", type=int, default=127)
    ap.add_argument("--norm_dbfs", type=float, default=-20.0)

    # FCAE
    ap.add_argument("--fc_ae", type=str, default="", help="Path to fc_ae.pt (optional)")
    ap.add_argument("--fc_ae_out", type=str, default="recon", choices=["recon","latent"],
                    help="Use reconstructed output or encoded latent from FCAE")
    ap.add_argument("--fcae_in_loop", action="store_true",
                    help="If set, apply FCAE inside the ascent loop (recommended for noticeable effect)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}+cu{torch.version.cuda} | CUDA={torch.cuda.is_available()} | device={device}")

    # Model + reference
    ldm = build_model(args.ckpt)

    try:
        ref_wav, _ = librosa.load(args.ref, sr=args.sr)
    except Exception as e:
        print(f"[combined] load failed ({e}); using 440 Hz tone", file=sys.stderr)
        t = np.linspace(0, args.duration, int(args.sr * args.duration), endpoint=False)
        ref_wav = 0.2 * np.sin(2*np.pi*440.0*t).astype(np.float32)
    ref_wave = torch.tensor(ref_wav, dtype=torch.float32, device=device).unsqueeze(0)
    ref_len_s = max(0.25, len(ref_wav) / float(args.sr))
    eff_duration = min(args.duration, ref_len_s)

    # Conditioning (this D defines UNet-facing latent size)
    with torch.no_grad():
        cond0 = encode_audio_cond(ldm, ref_wave, sr=args.sr)
    if not isinstance(cond0, torch.Tensor):
        cond0 = torch.zeros((1, 1, 513, 291), dtype=torch.float32, device=device)
    print(f"[cond] shape: {tuple(cond0.shape)} ; UNet latent dim D = {cond0.shape[-1]}")

    # Mel transformer & objective
    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels).to(device)
    obj = band_energy_objective(to_mel, band=(args.band_lo, args.band_hi))

    # ---------------- NORMAL (baseline, no ascent, no FCAE) ----------------
    normal_out = generate_with_audio_cond(
        ldm, cond0, steps=args.inner_steps, guidance_scale=args.guidance,
        duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr
    )
    mel_ref = ensure_mel_b1ft(to_mel(normalize_rms(ref_wave, target_dbfs=args.norm_dbfs)))
    mel_normal = to_mel_or_passthrough(normal_out, to_mel)

    # Save baseline images
    vmin_r, vmax_r = np.percentile(mel_ref[0,0].detach().cpu().numpy().T, [2, 98]).tolist()
    vmin_n, vmax_n = np.percentile(mel_normal[0,0].detach().cpu().numpy().T, [2, 98]).tolist()
    plot_mel_time_y(mel_ref, path="normal_original_norm.png", sr=args.sr, hop=args.hop,
                    title="Original (RMS-norm) — baseline", vmin=vmin_r, vmax=vmax_r)
    plot_mel_time_y(mel_normal, path="normal_output.png", sr=args.sr, hop=args.hop,
                    title="Model output — baseline", vmin=vmin_n, vmax=vmax_n)

    # ---------------- ASCENT (optionally with FCAE) ----------------
    e = torch.nn.Parameter(cond0.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([e], lr=args.lr)

    # FCAE setup
    use_fcae = bool(args.fc_ae)
    if use_fcae:
        fc_ae: FcAutoEncoder = FcAutoEncoder.load(args.fc_ae, map_location=device).to(device)
        fc_ae.train(False)  # we keep it frozen; gradients flow through it
        fc_in = int(fc_ae.cfg.in_dim)
        e_dim = int(e.shape[-1])
        print(f"[FCAE] in_dim={fc_in}, ascent_dim={e_dim}, in_loop={args.fcae_in_loop}")

        def fcae_transform(x: torch.Tensor) -> torch.Tensor:
            # x (..., e_dim) -> align to fc_in -> FCAE -> back to e_dim
            x_to = repeat_to_dim(x, fc_in)
            y = fc_ae.encode(x_to) if args.fc_ae_out == "latent" else fc_ae(x_to)
            return repeat_to_dim(y, e_dim)
    else:
        def fcae_transform(x: torch.Tensor) -> torch.Tensor:
            return x  # identity

    # Optimize e (FCAE optionally **inside** loop)
    for _ in tqdm(range(args.ascent_steps)):
        opt.zero_grad(set_to_none=True)
        e_used = fcae_transform(e) if args.fcae_in_loop else e

        syn = generate_with_audio_cond(
            ldm, e_used, steps=args.inner_steps, guidance_scale=args.guidance,
            duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr
        )
        score = obj(syn if syn.ndim > 1 else syn.unsqueeze(0))
        if score.ndim: score = score.mean()

        reg = 1e-3 * (e - cond0).pow(2).mean()
        loss = -(score) + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_([e], 5.0)
        opt.step()

        with torch.no_grad():
            d = e - cond0
            n = d.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            e.data = cond0 + d * (args.tau / n).clamp(max=1.0)

    # Final pass through FCAE (even if used in-loop, do a clean final transform)
    e_final = fcae_transform(e).detach()

    # Generate ascent output
    ascent_out = generate_with_audio_cond(
        ldm, e_final, steps=args.inner_steps, guidance_scale=args.guidance,
        duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr
    )
    mel_ascent = to_mel_or_passthrough(ascent_out, to_mel)

    # Diagnostics: show that FCAE/ascent changed things
    with torch.no_grad():
        diff_latent_mean = (e_final - cond0).abs().mean().item()
        diff_latent_vs_e = (e_final - e.detach()).abs().mean().item()
        # Compare spectrograms on a common size
        a = mel_normal[0,0].detach().cpu().numpy()
        b = mel_ascent[0,0].detach().cpu().numpy()
        # center-crop to min shape
        Fm, Tm = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        mse_out = float(np.mean((a[:Fm,:Tm] - b[:Fm,:Tm])**2))
    print(f"[diag] mean|e_final - cond0| = {diff_latent_mean:.6e}")
    print(f"[diag] mean|e_final - e|     = {diff_latent_vs_e:.6e}")
    print(f"[diag] MSE(normal_output, ascent_output) = {mse_out:.6e}")

    # Save ascent images
    vmin_a, vmax_a = np.percentile(mel_ascent[0,0].detach().cpu().numpy().T, [2, 98]).tolist()
    plot_mel_time_y(mel_ascent, path="ascent_output.png", sr=args.sr, hop=args.hop,
                    title=("Ascent + FCAE (swapped axes)" if use_fcae else "Ascent (swapped axes)"),
                    vmin=vmin_a, vmax=vmax_a)

    # Save tensors
    torch.save({
        "cond0": cond0.detach().cpu(),
        "e_final": e.detach().cpu(),
        "e_final_after_fcae": e_final.detach().cpu(),
        "mel_ref": mel_ref.cpu(),
        "mel_normal": mel_normal.cpu(),
        "mel_ascent": mel_ascent.cpu(),
    }, "combined_outputs.pt")

    print("Wrote:")
    print("  normal_original_norm.png, normal_output.png")
    print("  ascent_output.png")
    print("  combined_outputs.pt")
    print("If MSE(normal, ascent) is ~0, enable --fcae_in_loop and/or train FCAE on the SAME D as cond0.")
    print("Also try training FCAE with noise/regularization so it's not identity.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
