#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, warnings, sys, numpy as np, matplotlib.pyplot as plt, torch, librosa
from tqdm import tqdm

from sans import build_model
from sans.audio_utils import WaveToMel
from sans.objectives import band_energy_objective
from sans.ascent_ckpt import encode_audio_cond, generate_with_audio_cond
from sans.fc_autoencoder import FcAutoEncoder  # optional FCAE

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

# ---------------- unify mel/spec shapes to [B,1,F,T] ----------------
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

# ---------------- use WaveToMel only for wave; pass-through for specs ------------
def to_mel_or_passthrough(x: torch.Tensor, to_mel: torch.nn.Module) -> torch.Tensor:
    """
    If x is a waveform ([T], [B,T], [B,1,T]) -> compute mel with to_mel.
    If x is already spectrogram-like -> just coerce to [B,1, F, T].
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    # waveform heuristics
    if x.ndim == 1:
        return ensure_mel_b1ft(to_mel(x.unsqueeze(0)))
    if x.ndim == 2 and x.shape[0] <= 4:   # [B,T] (small B)
        return ensure_mel_b1ft(to_mel(x))
    if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] > 8:  # [B,1,T]
        return ensure_mel_b1ft(to_mel(x.squeeze(1)))
    # otherwise assume spectrogram-like (already [*,*,F,T]-ish)
    return ensure_mel_b1ft(x)

# ---------------- dimension adapter (pad or truncate) ----------------
def align_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Make x's last dimension == target_dim by zero-padding or truncation.
    Works for [B,D] or [B,T,D].
    """
    d = x.shape[-1]
    if d == target_dim:
        return x
    if d < target_dim:
        pad = target_dim - d
        pad_shape = list(x.shape[:-1]) + [pad]
        z = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, z], dim=-1)
    return x[..., :target_dim].contiguous()

# ---------------- plotting (SWAPPED axes: X=mel bins, Y=time) ----------------
def plot_mel_time_y(mel_b1ft: torch.Tensor, *, path: str, sr: int, hop: int, title: str = None, vmin=None, vmax=None):
    """
    Plot with X=mel bins, Y=time (seconds). Data plotted is img.T.
    """
    m = ensure_mel_b1ft(mel_b1ft).detach().cpu().numpy()
    img = m[0, 0]  # [F,T]
    F, T = img.shape
    t_y = (T - 1) * (hop / sr)
    data = img.T  # swap axes

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
    ap = argparse.ArgumentParser(description="Ascent with optional FCAE on latent (axes swapped: X=mel, Y=time)")
    ap.add_argument("--ckpt", type=str, default="/home/tori/.cache/audioldm/audioldm-s-full.ckpt")
    ap.add_argument("--ref", type=str, default="trumpet.wav")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--hop", type=int, default=80)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--ascent_steps", type=int, default=25)
    ap.add_argument("--inner_steps", type=int, default=12)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--band_lo", type=int, default=64)
    ap.add_argument("--band_hi", type=int, default=127)
    ap.add_argument("--norm_dbfs", type=float, default=-20.0)
    ap.add_argument("--fc_ae", type=str, default="", help="Path to fc_ae.pt (optional)")
    ap.add_argument("--fc_ae_out", type=str, default="recon", choices=["recon","latent"],
                    help="Use reconstructed output or encoded latent from FCAE")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}+cu{torch.version.cuda} | CUDA={torch.cuda.is_available()} | device={device}")

    # Model + reference audio
    ldm = build_model(args.ckpt)

    try:
        ref_wav, _ = librosa.load(args.ref, sr=args.sr)
    except Exception as e:
        print(f"[sample] load failed ({e}); using 440 Hz tone", file=sys.stderr)
        t = np.linspace(0, args.duration, int(args.sr * args.duration), endpoint=False)
        ref_wav = 0.2 * np.sin(2*np.pi*440.0*t).astype(np.float32)
    ref_wave = torch.tensor(ref_wav, dtype=torch.float32, device=device).unsqueeze(0)

    # Durations
    ref_len_s = max(0.25, len(ref_wav) / float(args.sr))
    eff_duration = min(args.duration, ref_len_s)

    # Conditioning from ckpt
    with torch.no_grad():
        cond0 = encode_audio_cond(ldm, ref_wave, sr=args.sr)
    if not isinstance(cond0, torch.Tensor):
        cond0 = torch.zeros((1, 768), dtype=torch.float32, device=device)

    # Objective (mel band energy)
    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels).to(device)
    obj = band_energy_objective(to_mel, band=(args.band_lo, args.band_hi))

    # Ascent variable
    e = torch.nn.Parameter(cond0.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([e], lr=args.lr)

    print(f"[shapes] cond0: {tuple(cond0.shape)}  (this sets ascent dim D={cond0.shape[-1]})")

    # Optimize e
    for _ in tqdm(range(args.ascent_steps)):
        opt.zero_grad(set_to_none=True)
        syn = generate_with_audio_cond(
            ldm, e, steps=args.inner_steps, guidance_scale=args.guidance,
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

    # ----- FCAE (optional) with auto dimension alignment -----
    e_ae = e
    if args.fc_ae:
        fc_ae: FcAutoEncoder = FcAutoEncoder.load(args.fc_ae, map_location=device).to(device).eval()
        fc_in = int(fc_ae.cfg.in_dim)
        e_dim = e.shape[-1]
        e_for_fc = align_last_dim(e, fc_in)
        with torch.no_grad():
            e_fc = fc_ae.encode(e_for_fc) if args.fc_ae_out == "latent" else fc_ae(e_for_fc)
        e_ae = align_last_dim(e_fc, e_dim).detach()

    # ----- Final generation using (possibly) FCAE-processed e -----
    out_wave = generate_with_audio_cond(
        ldm, e_ae, steps=args.inner_steps, guidance_scale=args.guidance,
        duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr
    )

    # Save mel previews (axes swapped: X=mel, Y=time)
    ref_wave_norm = normalize_rms(ref_wave, target_dbfs=args.norm_dbfs)
    mel_ref = ensure_mel_b1ft(to_mel(ref_wave_norm))
    mel_out = to_mel_or_passthrough(out_wave, to_mel)

    # per-image limits
    vmin_ref, vmax_ref = np.percentile(mel_ref[0,0].detach().cpu().numpy().T, [2, 98]).tolist()
    vmin_out, vmax_out = np.percentile(mel_out[0,0].detach().cpu().numpy().T, [2, 98]).tolist()

    plot_mel_time_y(mel_ref, path="original_norm.png", sr=args.sr, hop=args.hop,
                    title="Original (RMS-normalized)", vmin=vmin_ref, vmax=vmax_ref)
    plot_mel_time_y(mel_out, path="ascent_final.png", sr=args.sr, hop=args.hop,
                    title=("Ascent+FCAE (axes swapped)" if args.fc_ae else "Ascent (axes swapped)"),
                    vmin=vmin_out, vmax=vmax_out)

    # torch.save({
    #     "e_init": cond0.detach().cpu(),
    #     "e_final": e.detach().cpu(),
    #     "e_final_after_fcae": e_ae.detach().cpu() if isinstance(e_ae, torch.Tensor) else None,
    # }, "ascent_latents.pt")

    print("Wrote: original_norm.png, ascent_final.png, ascent_latents.pt (axes swapped)")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
