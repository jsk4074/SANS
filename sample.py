from sans import build_model
from sans.pipeline import style_transfer, build_diffusers, ascent_transfer
from sans.audio_utils import WaveToMel
from sans.objectives import make_ae_recon_objective, band_energy_objective

import matplotlib.pyplot as plt
import librosa
import numpy as np
import torch

def save_mel_image(
    mel: torch.Tensor,
    path: str = "mel.png",
    sr: int = 16000,          # match your repo/config
    hop_length: int = 160,    # match your repo/config (10 ms at 16 kHz)
    vmin: float = None,
    vmax: float = None,
    title: str = None,
):
    """Save a mel/fbank spectrogram tensor to an image."""
    # unwrap common shapes: [B,1,T,F], [B,F,T], [F,T], etc.
    if isinstance(mel, (list, tuple)):
        mel = mel[0]
    if isinstance(mel, torch.Tensor):
        mel = mel.detach().cpu()
    img = mel.numpy()

    # reduce dimensions
    if img.ndim == 4:                           # [B, C, T, F] (or [B, C, F, T])
        img = img[0, 0] if img.shape[1] == 1 else img[0]
    if img.ndim == 3:                           # [C, F, T] (take first channel)
        img = img[0]

    # robust display scaling
    if vmin is None or vmax is None:
        vmin = float(np.percentile(img, 2.0))
        vmax = float(np.percentile(img, 98.0))

    # time axis (seconds)
    n_frames = img.shape[1]
    times = np.arange(n_frames) * (hop_length / sr)

    plt.figure(figsize=(10, 3))
    plt.imshow(img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label("log-mel" )  # change if yours isn't log-scaled
    plt.xlabel("Time (s)")
    plt.ylabel("Mel bins")
    if title:
        plt.title(title)
    # optionally show reasonable time ticks
    xticks = np.linspace(0, n_frames - 1, 6)
    plt.xticks(xticks, [f"{t:.2f}" for t in (xticks * hop_length / sr)])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ldm = build_model("/home/tori/.cache/audioldm/audioldm-s-full.ckpt")  # or your checkpoint 

    # 2) (new) Diffusers pipeline for embedding ascent
    pipe = build_diffusers("cvssp/audioldm-s-full-v2", device=device)

    # 3) Define an anomaly objective (example: AE recon on mel)
    to_mel = WaveToMel(
        sr = 16000, 
        hop = 80, 
        n_mels = 128,
    )
    obj = band_energy_objective(
        to_mel, 
        band = (64, 127), 
        device = "cuda"
    )

    # 4) Run ascent (text can be empty like your sample)
    mel_ascent = ascent_transfer(
        pipe, "",
        objective_fn=obj,
        ascent_steps=25,
        inner_ddim_steps=12,
        guidance_scale=2.5,
        duration=5.0,
        tau=2.0,
        lr=1e-2,
        seed=1234,
        output_type="mel",
        to_mel=to_mel,
    )

    # 5) For comparison: your existing style_transfer path (unchanged)
    mel_style = style_transfer(
        ldm,
        "",
        original_audio_file_path="trumpet.wav",
        transfer_strength=0.3,
        duration=5.0,
        output_type="mel",
    )

    # 6) Visualize
    org, sr = librosa.load("./trumpet.wav", sr=16000)
    org = librosa.feature.melspectrogram(y=org, sr=16000, hop_length=80, n_mels=128)
    org = torch.tensor(org)

    print(org.size())
    print(mel_style.size())
    print(mel_ascent.size())

    save_mel_image(org, "original.png", sr=16000, hop_length=80, title="Original")
    save_mel_image(mel_style, "src_style.png", sr=16000, hop_length=80, title="Style transfer")
    save_mel_image(mel_ascent, "src_ascent.png", sr=16000, hop_length=80, title="Ascent synthesis")

if __name__ == "__main__": main()