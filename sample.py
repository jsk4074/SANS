from sans import build_model
from sans.pipeline import style_transfer

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


ldm = build_model("/home/tori/.cache/audioldm/audioldm-s-full.ckpt")  # or your checkpoint 

# 2) Style transfer → spectrogram 
mel = style_transfer(
    ldm, 
    "",                  
    original_audio_file_path="trumpet.wav", 
    transfer_strength=0.3, 
    duration=5.0, 
    output_type="mel"
)

org, sr = librosa.load("./trumpet.wav")
org = librosa.feature.melspectrogram(y = org)
org = torch.tensor(org)
print(org.size())
print(mel.size())
save_mel_image(org, "original.png", sr=16000, hop_length=80)
save_mel_image(mel, "src.png", sr=16000, hop_length=80)
