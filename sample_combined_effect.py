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

# ---------------- differentiable adapters (repeat/slice) ------------------------
def repeat_to_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    d = int(x.shape[-1])
    if d == target_dim: return x
    if d > target_dim:  return x[..., :target_dim].contiguous()
    reps = (target_dim + d - 1) // d
    x_rep = x.repeat_interleave(reps, dim=-1)
    return x_rep[..., :target_dim].contiguous()

def fcae_extrapolate(e: torch.Tensor, fc_ae: FcAutoEncoder, e_dim: int, fc_in: int,
                     alpha: float, use_latent: bool) -> torch.Tensor:
    """
    e_used = e + alpha * (FCAE(e_to_in) - e)
    """
    e_to = repeat_to_dim(e, fc_in)
    y = fc_ae.encode(e_to) if use_latent else fc_ae(e_to)
    y_back = repeat_to_dim(y, e_dim)
    return e + alpha * (y_back - e)

# ---------------- plotting (SWAPPED axes: X=mel, Y=time) -----------------------
def plot_mel_time_y(mel_b1ft: torch.Tensor, *, path: str, sr: int, hop: int, title: str = None, vmin=None, vmax=None):
    m = ensure_mel_b1ft(mel_b1ft).detach().cpu().numpy()
    img = m[0, 0]
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
    ap = argparse.ArgumentParser(description="Combined baseline + ascent with FCAE and ablation probes (axes swapped)")
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
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--band_lo", type=int, default=64)
    ap.add_argument("--band_hi", type=int, default=127)
    ap.add_argument("--norm_dbfs", type=float, default=-20.0)

    # FCAE
    ap.add_argument("--fc_ae", type=str, default="")
    ap.add_argument("--fc_ae_out", type=str, default="recon", choices=["recon","latent"])
    ap.add_argument("--fcae_in_loop", action="store_true")
    ap.add_argument("--fcae_alpha", type=float, default=1.0, help="Extrapolation strength; >1 exaggerates FCAE effect")
    ap.add_argument("--fcae_dropout_eval", action="store_true", help="Keep FCAE dropout active at inference")

    # Probes to see if conditioning matters
    ap.add_argument("--probe_cond", action="store_true",
                    help="Generate with zero/random/scaled conditions and report MSE vs baseline")
    ap.add_argument("--probe_scale", type=float, default=2.0, help="Scale factor for probe 'scaled' condition")
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
    print(f"[cond] shape: {tuple(cond0.shape)} ; latent D = {cond0.shape[-1]}")

    # Mel transformer & objective
    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels).to(device)
    obj = band_energy_objective(to_mel, band=(args.band_lo, args.band_hi))

    # ---------------- NORMAL (baseline, no ascent) ----------------
    def gen_with(cond: torch.Tensor):
        return generate_with_audio_cond(
            ldm, cond, steps=args.inner_steps, guidance_scale=args.guidance,
            duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr
        )

    normal_out = gen_with(cond0)
    mel_ref = ensure_mel_b1ft(to_mel(normalize_rms(ref_wave, target_dbfs=args.norm_dbfs)))
    mel_normal = to_mel_or_passthrough(normal_out, to_mel)

    vmin_r, vmax_r = np.percentile(mel_ref[0,0].detach().cpu().numpy().T, [2, 98]).tolist()
    vmin_n, vmax_n = np.percentile(mel_normal[0,0].detach().cpu().numpy().T, [2, 98]).tolist()
    plot_mel_time_y(mel_ref, path="normal_original_norm.png", sr=args.sr, hop=args.hop,
                    title="Original (RMS-norm) — baseline", vmin=vmin_r, vmax=vmax_r)
    plot_mel_time_y(mel_normal, path="normal_output.png", sr=args.sr, hop=args.hop,
                    title="Model output — baseline", vmin=vmin_n, vmax=vmax_n)

    # ---------------- PROBE: does the model even use the condition? --------------
    def mse_between(a: torch.Tensor, b: torch.Tensor) -> float:
        A = to_mel_or_passthrough(a, to_mel)[0,0].detach().cpu().numpy()
        B = to_mel_or_passthrough(b, to_mel)[0,0].detach().cpu().numpy()
        Fm, Tm = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
        return float(np.mean((A[:Fm,:Tm] - B[:Fm,:Tm])**2))

    if args.probe_cond:
        zero_cond = torch.zeros_like(cond0)
        rand_cond = torch.randn_like(cond0)
        scaled_cond = cond0 * args.probe_scale

        out_zero   = gen_with(zero_cond)
        out_rand   = gen_with(rand_cond)
        out_scaled = gen_with(scaled_cond)

        print("[probe] MSE(normal, zero_cond)   =", f"{mse_between(normal_out, out_zero):.6e}")
        print("[probe] MSE(normal, random_cond) =", f"{mse_between(normal_out, out_rand):.6e}")
        print("[probe] MSE(normal, scaled_cond) =", f"{mse_between(normal_out, out_scaled):.6e}")
        print("If these are ~0, your generator is ignoring the conditioning input entirely.")

    # ---------------- ASCENT with optional FCAE and EXTRAPOLATION ----------------
    e = torch.nn.Parameter(cond0.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([e], lr=args.lr)

    use_fcae = bool(args.fc_ae)
    if use_fcae:
        fc_ae: FcAutoEncoder = FcAutoEncoder.load(args.fc_ae, map_location=device).to(device)
        if args.fcae_dropout_eval:
            fc_ae.train(True)   # keep dropout active to avoid exact identity
        else:
            fc_ae.train(False)
        fc_in = int(fc_ae.cfg.in_dim)
        e_dim = int(e.shape[-1])
        use_latent = (args.fc_ae_out == "latent")
        print(f"[FCAE] in_dim={fc_in}, ascent_dim={e_dim}, in_loop={args.fcae_in_loop}, alpha={args.fcae_alpha}, dropout_eval={args.fcae_dropout_eval}")

        def xform(x: torch.Tensor) -> torch.Tensor:
            # Extrapolated FCAE effect
            return fcae_extrapolate(x, fc_ae, e_dim=e_dim, fc_in=fc_in, alpha=args.fcae_alpha, use_latent=use_latent)
    else:
        def xform(x: torch.Tensor) -> torch.Tensor:
            return x

    # Optimize e
    for _ in tqdm(range(args.ascent_steps)):
        opt.zero_grad(set_to_none=True)
        e_used = xform(e) if args.fcae_in_loop else e
        syn = gen_with(e_used)
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

    # Final transform and generation
    e_final = xform(e).detach()
    ascent_out = gen_with(e_final)
    mel_ascent = to_mel_or_passthrough(ascent_out, to_mel)

    # Diagnostics
    with torch.no_grad():
        diff_latent_mean = (e_final - cond0).abs().mean().item()
        diff_latent_vs_e = (e_final - e.detach()).abs().mean().item()
        mse_out = mse_between(normal_out, ascent_out)
    print(f"[diag] mean|e_final - cond0| = {diff_latent_mean:.6e}")
    print(f"[diag] mean|e_final - e|     = {diff_latent_vs_e:.6e}")
    print(f"[diag] MSE(normal_output, ascent_output) = {mse_out:.6e}")

    vmin_a, vmax_a = np.percentile(mel_ascent[0,0].detach().cpu().numpy().T, [2, 98]).tolist()
    plot_mel_time_y(mel_ascent, path="ascent_output.png", sr=args.sr, hop=args.hop,
                    title=("Ascent + FCAE (swapped axes)" if use_fcae else "Ascent (swapped axes)"),
                    vmin=vmin_a, vmax=vmax_a)

    # Save tensors
    torch.save({
        "cond0": cond0.detach().cpu(),
        "e_final": e.detach().cpu(),
        "e_final_after_xform": e_final.detach().cpu(),
        "mel_ref": mel_ref.cpu(),
        "mel_normal": mel_normal.cpu(),
        "mel_ascent": mel_ascent.cpu(),
    }, "combined_outputs.pt")

    print("Wrote: normal_original_norm.png, normal_output.png, ascent_output.png, combined_outputs.pt")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
