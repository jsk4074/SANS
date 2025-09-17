# sans/objectives.py
# -*- coding: utf-8 -*-

from typing import Callable, Tuple
import sys
import torch


def _safe_log(msg: str):
    print(f"[objectives] {msg}", file=sys.stderr)


def _ensure_bft_from_any(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize to [B,F,T] from any of:
      [B,1,F,T], [B,F,T], [F,T]
    """
    if x.ndim == 4:  # [B,C,F,T]
        x = x[:, 0] if x.size(1) == 1 else x.mean(dim=1)
    elif x.ndim == 3:  # [B,F,T]
        pass
    elif x.ndim == 2:  # [F,T]
        x = x.unsqueeze(0)
    else:
        # As a last resort, make a fake mel-like map
        _safe_log(f"_ensure_bft_from_any: unexpected shape {tuple(x.shape)}, using zeros.")
        B = x.shape[0] if x.ndim > 0 else 1
        x = torch.zeros((B, 128, 32), dtype=x.dtype, device=x.device)
    return x


def _looks_like_mel(x: torch.Tensor, n_fft_hint: int = 1024) -> bool:
    """
    Heuristic: treat as mel/spectrogram if it already has a frequency axis.
    We use: has >=2 dims AND second-to-last dim is between 32..2048 (F),
    and time dim not huge (<= 8192).
    """
    if not isinstance(x, torch.Tensor):
        return False
    if x.ndim < 2:
        return False
    F = x.shape[-2]
    T = x.shape[-1]
    return (32 <= F <= 2048) and (T <= 8192)


def band_energy_objective(
    to_mel: torch.nn.Module,
    band: Tuple[int, int] = (64, 127),
) -> Callable[[torch.Tensor], torch.Tensor]:
    lo_req, hi_req = band

    def _obj(x: torch.Tensor) -> torch.Tensor:
        # If it's already mel-like, use it directly; else compute mel from waveform
        if _looks_like_mel(x):
            mel = _ensure_bft_from_any(x)
        else:
            if hasattr(to_mel, "to"):
                to_mel.to(x.device)
            mel = to_mel(x)  # returns [B,1,F,T]
            mel = _ensure_bft_from_any(mel)

        B, F, T = mel.shape
        lo = max(0, min(int(lo_req), F - 1))
        hi = max(lo, min(int(hi_req), F - 1))
        mel_band = mel[:, lo : hi + 1, :]
        if mel_band.numel() == 0:
            mel_band = mel
        return mel_band.abs().mean()

    return _obj
