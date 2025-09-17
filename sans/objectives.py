# sans/objectives.py
# -*- coding: utf-8 -*-
"""
Lightweight objectives for SANS synthesis loops.

Exports
-------
band_energy_objective(to_mel, band=(64, 127))
    Returns a closure objective(wave) that maximizes average mel energy
    inside the given [lo, hi] mel bin band (inclusive).

make_ae_recon_objective(ae_model, to_mel)
    Returns a closure objective(wave) that maximizes AE reconstruction error
    (L1 by default) on the mel features produced by `to_mel`.
"""

from typing import Callable, Tuple
import torch
import torch.nn.functional as F


__all__ = [
    "band_energy_objective",
    "make_ae_recon_objective",
]


def _ensure_bft(mel: torch.Tensor) -> torch.Tensor:
    """
    Normalize mel tensor shapes to [B, F, T].

    Accepts:
      [B, 1, F, T]  -> [B, F, T]
      [B, C, F, T]  -> average across C -> [B, F, T]
      [B, F, T]     -> [B, F, T]
      [F, T]        -> [1, F, T]
    """
    if mel.ndim == 4:
        # [B, C, F, T]
        if mel.size(1) == 1:
            mel = mel[:, 0]
        else:
            mel = mel.mean(dim=1)
    elif mel.ndim == 2:
        mel = mel.unsqueeze(0)
    elif mel.ndim != 3:
        raise ValueError(f"Unexpected mel shape: {tuple(mel.shape)}")
    return mel


def band_energy_objective(
    to_mel: torch.nn.Module,
    band: Tuple[int, int] = (64, 127),
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a simple, fully differentiable objective that *maximizes*
    average energy in a target mel band.

    Parameters
    ----------
    to_mel : nn.Module
        Module that maps waveform -> mel. Should accept [B, T] or [B, 1, T]
        and return [B, F, T] or [B, 1, F, T]. It will be moved to the
        waveform's device if possible.
    band : (int, int)
        (lo, hi) mel bin indices, inclusive. Values are clamped to valid range
        after seeing the actual mel size.

    Returns
    -------
    objective : Callable[[Tensor], Tensor]
        A closure that takes waveform Tensor [B, T] (or [B, 1, T]) and
        returns a scalar Tensor to maximize.
    """
    lo_req, hi_req = band

    def _obj(wave: torch.Tensor) -> torch.Tensor:
        # Try to run to_mel on the same device as wave for speed; if the
        # module doesn't support that device, fall back gracefully to CPU.
        try:
            to_mel_dev = next(to_mel.parameters(), None)
            # Some transforms have no parameters; still try moving buffers
            if hasattr(to_mel, "to"):
                to_mel.to(wave.device)
            mel = to_mel(wave)
        except Exception:
            # Fallback: compute on CPU and move result back (still differentiable).
            mel = to_mel.to("cpu")(wave.to("cpu")).to(wave.device)

        mel = _ensure_bft(mel)  # [B, F, T]
        B, F, T = mel.shape

        lo = max(0, min(int(lo_req), F - 1))
        hi = max(lo, min(int(hi_req), F - 1))  # inclusive
        mel_band = mel[:, lo : hi + 1, :]      # [B, F_sel, T]

        # If something went wrong and the slice is empty, fall back to global mean
        if mel_band.numel() == 0:
            mel_band = mel

        # Maximize average magnitude (works with dB-scaled or linear mels)
        # Mean over F and T; if you want per-example mean, drop the final .mean().
        score = mel_band.abs().mean()

        return score

    return _obj


def make_ae_recon_objective(
    ae_model: torch.nn.Module,
    to_mel: torch.nn.Module,
    reduction: str = "mean",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create an objective that *maximizes* AE reconstruction error on mel.

    Parameters
    ----------
    ae_model : nn.Module
        Autoencoder (expects mel-like tensor). Should map [B,1,F,T] or [B,F,T]
        to a tensor of the same shape.
    to_mel : nn.Module
        Wave -> mel transform (see notes in band_energy_objective).
    reduction : str
        'mean' or 'sum' over batch. The inner spatial reduction is always mean.

    Returns
    -------
    objective : Callable[[Tensor], Tensor]
        Closure to maximize AE reconstruction error (L1).
    """
    assert reduction in ("mean", "sum")

    def _obj(wave: torch.Tensor) -> torch.Tensor:
        # Align devices (same strategy as above).
        try:
            if hasattr(to_mel, "to"):
                to_mel.to(wave.device)
            mel = to_mel(wave)
        except Exception:
            mel = to_mel.to("cpu")(wave.to("cpu")).to(wave.device)

        mel = _ensure_bft(mel)  # [B, F, T]
        # AE commonly expects [B,1,F,T]
        mel_in = mel.unsqueeze(1)  # [B,1,F,T]

        # Ensure AE is on the right device
        if hasattr(ae_model, "to"):
            ae_model.to(mel_in.device)

        recon = ae_model(mel_in)

        # L1 reconstruction error over spatial dims -> [B]
        err = F.l1_loss(recon, mel_in, reduction="none")
        while err.ndim > 1:
            err = err.mean(dim=-1)
        # err now shape [B]

        return err.mean() if reduction == "mean" else err.sum()

    return _obj
