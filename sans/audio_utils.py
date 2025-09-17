# sans/audio_utils.py
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn.functional as F
from typing import Optional


def _power_to_db(S: torch.Tensor, top_db: Optional[float] = 80.0) -> torch.Tensor:
    """
    Convert a power spectrogram to decibel units.
    - S: power spectrogram (>=0), shape [B, 1, F, T]
    - top_db: clip the dynamic range to [-top_db, 0] dB relative to each sample's max
    Returns: log-power in dB, same shape.
    """
    S = S.clamp_min(1e-10)
    # reference is per-sample global max
    ref = S.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
    log_spec = 10.0 * torch.log10(S) - 10.0 * torch.log10(ref)
    if top_db is not None:
        log_spec = torch.clamp(log_spec, min=-float(top_db))
    return log_spec


class WaveToSpec(torch.nn.Module):
    """
    Standard STFT spectrogram (linear frequency).
    Output: [B, 1, F, T] (power or power-dB).
    """
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 1024,
        hop: int = 160,
        win_length: Optional[int] = None,
        center: bool = False,          # keep False to avoid huge padding surprises
        power: float = 2.0,            # 1.0 -> magnitude, 2.0 -> power
        to_db: bool = True,            # return dB if True
        top_db: Optional[float] = 80.0 # clamp floor to -top_db dB relative to max
    ):
        super().__init__()
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.center = bool(center)
        self.power = float(power)
        self.to_db = bool(to_db)
        self.top_db = top_db

    def _pad_if_short(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        if T < self.n_fft:
            pad = self.n_fft - T
            x = F.pad(x, (0, pad), mode="constant", value=0.0)
        return x

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # Expect [B,T] or [B,1,T]; Return [B,1,F,Tf]
        x = wave if isinstance(wave, torch.Tensor) else torch.tensor(wave, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B,1,T]
        x = self._pad_if_short(x)

        window = torch.hann_window(self.win_length, device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=window,
            center=self.center,
            return_complex=True,
        )  # [B, F, Tf] complex
        mag = spec.abs()  # magnitude
        if self.power != 1.0:
            mag = mag ** self.power  # power if power=2.0

        out = mag.unsqueeze(1)  # [B,1,F,Tf]
        if self.to_db:
            out = _power_to_db(out, top_db=self.top_db)
        return out


class WaveToMel(torch.nn.Module):
    """
    Mel-like spectrogram using STFT + simple mel filter approximation is omitted here
    to keep things robust and pure torch. If you already have a torchaudio version
    that works for you, keep it; otherwise this naive version mimics log-mel by
    applying log1p to magnitude.
    Output: [B,1,F,T]
    """
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 1024,
        hop: int = 160,
        n_mels: int = 128,
        power: float = 2.0,
    ):
        super().__init__()
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.n_mels = int(n_mels)
        self.power = float(power)

    def _pad_if_short(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        if T < self.n_fft:
            pad = self.n_fft - T
            x = F.pad(x, (0, pad), mode="constant", value=0.0)
        return x

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        x = wave if isinstance(wave, torch.Tensor) else torch.tensor(wave, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B,1,T]
        x = self._pad_if_short(x)

        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=window,
            center=False,
            return_complex=True,
        )  # [B,F,T]
        mag = spec.abs().unsqueeze(1)  # [B,1,F,T]
        mel_like = torch.log1p(mag)    # simple log compression (mel-ish)
        return mel_like
