# sans/audio_utils.py
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn.functional as F

try:
    import torchaudio as ta
    _HAVE_TA = True
except Exception as e:
    print(f"[audio_utils] torchaudio import failed: {e}. Using fallback STFT.", file=sys.stderr)
    _HAVE_TA = False


class WaveToMel(torch.nn.Module):
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 1024,
        hop: int = 160,
        n_mels: int = 128,
        fmin: int = 20,
        fmax: int = 8000,
        power: float = 2.0,
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.n_mels = int(n_mels)
        self.fmin = int(fmin)
        self.fmax = int(fmax)
        self.power = float(power)

        if _HAVE_TA:
            try:
                # center=False to avoid huge padding; we will explicit-pad if too short
                self.melspec = ta.transforms.MelSpectrogram(
                    sample_rate=sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop,
                    f_min=self.fmin,
                    f_max=self.fmax,
                    n_mels=self.n_mels,
                    center=False,
                    power=self.power,
                )
                self.amp2db = ta.transforms.AmplitudeToDB(stype="power")
            except Exception as e:
                print(f"[audio_utils] torchaudio transforms init failed: {e}. Using fallback STFT.", file=sys.stderr)
                self.melspec = None
                self.amp2db = None
        else:
            self.melspec = None
            self.amp2db = None

    def _pad_if_short(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure last dim >= n_fft so transforms don't crash. Zero-pad if needed.
        """
        T = x.shape[-1]
        if T < self.n_fft:
            pad = self.n_fft - T
            x = F.pad(x, (0, pad), mode="constant", value=0.0)
        return x

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # Expect waveform as [B,T] or [B,1,T]; return [B,1,F,Tf]
        x = wave
        if isinstance(x, torch.Tensor) and x.ndim == 2:
            x = x.unsqueeze(1)
        x = self._pad_if_short(x)

        # torchaudio path
        if _HAVE_TA and self.melspec is not None:
            mel = self.melspec(x)
            if self.amp2db is not None:
                mel = self.amp2db(mel)
            return mel

        # Fallback: STFT -> log magnitude
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        spec = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop,
                          window=window, return_complex=True, center=False)
        mag = spec.abs().unsqueeze(1)  # [B,1,F,Tf]
        mel_like = torch.log1p(mag)    # simple log scaling
        return mel_like
