#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, warnings, sys, numpy as np, matplotlib.pyplot as plt, torch, librosa
from tqdm import tqdm

from sans import build_model
from sans.audio_utils import WaveToMel
from sans.objectives import band_energy_objective
from sans.ascent_ckpt import encode_audio_cond, generate_with_audio_cond
from sans.pipeline import style_transfer

# ---------------- audio normalization (ONLY for original) ----------------
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
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    t = x
    if t.ndim == 4:                    # [B,1,F,T] or [B,C,F,T]
        if t.size(1) != 1: t = t.mean(dim=1, keepdim=True)
        return t
    if t.ndim == 3:                    # [B,F,T] or [B,T,F]
        B, A, B2 = t.shape
        return t.unsqueeze(1) if A <= B2 else t.transpose(1, 2).unsqueeze(1)
    if t.ndim == 2:                    # [F,T] or [T,F]
        F, T = t.shape
        return t.unsqueeze(0).unsqueeze(0) if F <= T else t.t().unsqueeze(0).unsqueeze(0)
    if t.ndim == 1:
        return t.unsqueeze(0).unsqueeze(0)
    raise RuntimeError(f"Unsupported tensor shape for ensure_mel_b1ft: {tuple(t.shape)}")

# ---------------- STFT (normal spectrogram) helpers ----------------
def _power_to_db(S: torch.Tensor, top_db: float = 80.0) -> torch.Tensor:
    # S: [B,1,F,T] power >= 0  ->  dB relative to per-sample max, clipped to [-top_db, 0]
    S = S.clamp_min(1e-10)
    ref = S.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
    log_spec = 10.0 * torch.log10(S) - 10.0 * torch.log10(ref)
    if top_db is not None:
        log_spec = torch.clamp(log_spec, min=-float(top_db))
    return log_spec

def wave_to_spec(
    wave: torch.Tensor, *, n_fft: int = 1024, hop: int = 160, win_length: int | None = None,
    power: float = 2.0, to_db: bool = True, top_db: float = 80.0, center: bool = False
) -> torch.Tensor:
    """
    PyTorch STFT → magnitude^power → (optional) dB. Returns [B,1,F,T].
    """
    x = wave if isinstance(wave, torch.Tensor) else torch.tensor(wave, dtype=torch.float32)
    if x.ndim == 2: x = x.unsqueeze(1)         # [B,1,T]
    if x.shape[-1] < n_fft:
        pad = n_fft - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, pad), mode="constant", value=0.0)

    win_len = int(n_fft) if win_length is None else int(win_length)
    window = torch.hann_window(win_len, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win_len,
        window=window, center=center, return_complex=True
    )                       # [B,F,T] complex
    mag = spec.abs()
    if power != 1.0: mag = mag ** power
    out = mag.unsqueeze(1)  # [B,1,F,T]
    if to_db:
        out = _power_to_db(out, top_db=top_db)
    return out

def ensure_spec_b1ft(x: torch.Tensor) -> torch.Tensor:
    # same coercion as mel helper, named for clarity
    return ensure_mel_b1ft(x)

# ---------------- plotting with explicit/auto axes ----------------
def plot_mel_auto(
    mel_b1ft: torch.Tensor, *, path: str, sr: int, hop: int,
    expected_sec: float | None, mode: str = "auto", title: str = None,
    vmin=None, vmax=None
):
    """
    mel_b1ft: [B,1,F,T]
    mode: "auto" (pick orientation so time range ~ expected_sec),
          "time-x" (X=time, Y=mel),
          "time-y" (X=mel, Y=time)
    """
    m = mel_b1ft.detach().cpu().numpy()
    if m.ndim != 4 or m.shape[1] != 1:
        m = ensure_mel_b1ft(torch.tensor(m)).numpy()
    img = m[0, 0]  # [F,T]
    F, T = img.shape

    # candidates
    t_x = (T - 1) * (hop / sr)  # time if we put time on X (use img)
    t_y = (F - 1) * (hop / sr)  # time if we put time on Y (use img.T)

    # choose orientation
    if mode == "auto" and expected_sec is not None:
        pick = "time-x" if abs(t_x - expected_sec) <= abs(t_y - expected_sec) else "time-y"
    else:
        pick = mode

    # intensities
    data = img if pick == "time-x" else img.T
    if vmin is None or vmax is None:
        vmin = float(np.percentile(data, 2.0)); vmax = float(np.percentile(data, 98.0))

    # extents
    if pick == "time-x":
        extent = [0.0, t_x, 0.0, float(F - 1)]
        xlabel, ylabel = "Time (s)", "Mel bins"
    else:
        extent = [0.0, float(F - 1), 0.0, t_y]
        xlabel, ylabel = "Mel bins", "Time (s)"

    plt.figure(figsize=(10, 4))
    plt.imshow(data, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(); cbar.set_label("log-mel")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_spec_time_x(spec_b1ft: torch.Tensor, *, path: str, sr: int, hop: int, title: str = None, vmin=None, vmax=None):
    """
    Plot STFT spectrogram with X=time (s), Y=frequency bins.
    """
    s = spec_b1ft.detach().cpu().numpy()
    if s.ndim != 4 or s.shape[1] != 1:
        s = ensure_spec_b1ft(torch.tensor(s)).numpy()
    img = s[0, 0]  # [F,T]
    F, T = img.shape
    t_x = (T - 1) * (hop / sr)

    if vmin is None or vmax is None:
        vmin = float(np.percentile(img, 2.0)); vmax = float(np.percentile(img, 98.0))

    extent = [0.0, t_x, 0.0, float(F - 1)]
    plt.figure(figsize=(10, 4))
    plt.imshow(img, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(); cbar.set_label("dB (rel. max)")
    plt.xlabel("Time (s)"); plt.ylabel("Frequency bins")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="CKPT-only ascent; normalize ONLY original; auto axes per image; plus STFT")
    ap.add_argument("--ckpt", type=str, default="/home/tori/.cache/audioldm/audioldm-s-full.ckpt")
    ap.add_argument("--ref", type=str, default="trumpet.wav")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=5.)
    ap.add_argument("--hop", type=int, default=10)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--ascent_steps", type=int, default=50)
    ap.add_argument("--inner_steps", type=int, default=12)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--band_lo", type=int, default=256)
    ap.add_argument("--band_hi", type=int, default=127)
    ap.add_argument("--norm_dbfs", type=float, default=-20.0, help="RMS target for ORIGINAL ONLY")
    ap.add_argument("--axes", type=str, default="auto", choices=["auto","time-x","time-y"],
                    help="auto picks orientation so time axis matches expected duration (mel plots)")
    ap.add_argument("--spec_nfft", type=int, default=1024, help="STFT n_fft for normal spectrogram")
    ap.add_argument("--save_numpy", action="store_true", help="Also save .npy alongside .pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}+cu{torch.version.cuda} | CUDA={torch.cuda.is_available()} | device={device}")

    # Model
    ldm = build_model(args.ckpt)

    # Reference audio
    try:
        ref_wav, _ = librosa.load(args.ref, sr=args.sr)
    except Exception as e:
        print(f"[sample] load failed ({e}); using 440Hz tone", file=sys.stderr)
        t = np.linspace(0, args.duration, int(args.sr * args.duration), endpoint=False)
        ref_wav = 0.2 * np.sin(2*np.pi*440.0*t).astype(np.float32)
    ref_wave = torch.tensor(ref_wav, dtype=torch.float32, device=device).unsqueeze(0)

    # Expected durations (used for auto axis picking)
    ref_len_s = max(0.25, len(ref_wav) / float(args.sr))
    eff_duration = min(args.duration, ref_len_s)

    # Conditioning
    with torch.no_grad():
        cond0 = encode_audio_cond(ldm, ref_wave, sr=args.sr)
    if not isinstance(cond0, torch.Tensor):
        cond0 = torch.zeros((1, 768), dtype=torch.float32, device=device)

    # Objective & mel transform
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

    # ----- Build MELs for comparison -----
    ref_wave_norm = normalize_rms(ref_wave, target_dbfs=args.norm_dbfs)
    to_mel.to(ref_wave_norm.device)
    mel_ref = ensure_mel_b1ft(to_mel(ref_wave_norm))    # [B,1,F,T]

    # Ascent: as-is
    if isinstance(last_out, torch.Tensor) and last_out.ndim == 2 and last_out.size(1) > 4*to_mel.n_fft:
        to_mel.to(last_out.device)
        mel_asc = ensure_mel_b1ft(to_mel(last_out))
        asc_waveform = last_out
    else:
        mel_asc = ensure_mel_b1ft(last_out)
        asc_waveform = None  # not a waveform

    # Per-image color scaling for MEL
    ref_img = mel_ref[0,0].detach().cpu().numpy()
    asc_img = mel_asc[0,0].detach().cpu().numpy()
    vmin_ref, vmax_ref = np.percentile(ref_img, [2,98]).tolist()
    vmin_asc, vmax_asc = np.percentile(asc_img, [2,98]).tolist()

    # MEL Plots 
    plot_mel_auto( 
        mel_ref,
        path="original_norm.png", 
        sr=args.sr, hop=args.hop, 
        expected_sec=ref_len_s, mode=args.axes, 
        title="Original (RMS-normalized)", 
        vmin=vmin_ref, vmax=vmax_ref 
    ) 
    plot_mel_auto( 
        mel_asc,
        path="ascent.png",
        sr=args.sr, hop=args.hop,
        expected_sec=eff_duration, mode="time-y",   # force X=time for ascent
        title="Ascent (as-is, time on X)",
        vmin=vmin_asc, vmax=vmax_asc
    )

    # ----- STFT (normal spectrograms) -----
    # Original (normalized) -> STFT

    spec_ref = style_transfer(
        ldm,
        "",  
        original_audio_file_path = args.ref,
        transfer_strength = 0.0,
        duration = args.duration,
        output_type = "waveform",
        ddim_steps = args.ascent_steps,
        guidance_scale = args.guidance,
        seed = 7777,
    )

    print(mel_ref.shape)
    print(mel_ref.shape)
    print(mel_ref.shape)
    print(mel_asc.shape)
    print(mel_asc.shape)
    print(mel_asc.shape)
    print(spec_ref.shape)
    print(spec_ref.shape)
    print(spec_ref.shape)

    # plot_spec_time_x(
    #     spec_ref, 
    #     path="original_norm_spec.png", 
    #     sr=args.sr, 
    #     hop=args.hop,
    #     title="Original (RMS-normalized) – STFT"
    # )
    plot_mel_auto( 
        spec_ref,
        path="original_norm_spec.png",
        sr=args.sr, hop=args.hop,
        expected_sec=eff_duration, mode="time-y",   # force X=time for ascent
        title="Ascent (as-is, time on X)",
        vmin=vmin_asc, vmax=vmax_asc
    )

    print(
        "Done. Files written:\n"
        "original_norm.png, ascent.png\n"
        "ascent_spec.png (if waveform)"
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
