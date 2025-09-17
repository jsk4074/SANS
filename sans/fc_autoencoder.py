# sans/fc_autoencoder.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class FcAeConfig:
    in_dim: int
    bottleneck: int = 256
    hidden_mult: float = 2.0   # hidden size = int(bottleneck * hidden_mult)
    depth: int = 2             # number of hidden layers in each of encoder/decoder
    dropout: float = 0.0
    activation: str = "gelu"   # "relu" | "gelu" | "silu"
    layernorm: bool = True


def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    return nn.GELU()  # default


def _align_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Make x's last dimension == target_dim by zero-padding or truncation.
    Works for any rank >= 1. No gradients are harmed :)
    """
    d = int(x.shape[-1])
    if d == target_dim:
        return x
    if d < target_dim:
        pad = target_dim - d
        pad_shape = (*x.shape[:-1], pad)
        z = x.new_zeros(pad_shape)
        return torch.cat([x, z], dim=-1)
    # d > target_dim
    return x[..., :target_dim].contiguous()


def _flatten_last(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
    """
    Accept any tensor with at least 1 dim: [..., D]
    Returns:
      x2d: [N, D]  (N = prod(leading_shape))
      leading_shape: tuple([...])
      D: last-dimension size before alignment
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 0:
        # scalar -> treat as length-1 vector
        x = x.view(1)
    leading_shape: Tuple[int, ...] = tuple(x.shape[:-1]) if x.ndim > 1 else (1,)
    D = int(x.shape[-1]) if x.ndim >= 1 else 1
    N = int(torch.tensor(leading_shape).prod().item()) if len(leading_shape) > 0 else 1
    x2d = x.reshape(N, D)
    return x2d, leading_shape, D


def _unflatten_last(x2d: torch.Tensor, leading_shape: Sequence[int]) -> torch.Tensor:
    """
    Inverse of _flatten_last for outputs whose last dim is already correct.
    """
    return x2d.view(*leading_shape, x2d.shape[-1])


class FcAutoEncoder(nn.Module):
    """
    Fully-connected autoencoder operating on the LAST dim.
    Accepts tensors shaped [..., D] and returns the same shape with last dim = in_dim (for forward)
    or = bottleneck (for encode).
    """
    def __init__(self, cfg: FcAeConfig):
        super().__init__()
        self.cfg = cfg
        H = max(8, int(round(cfg.bottleneck * cfg.hidden_mult)))
        act = _act(cfg.activation)

        enc_layers: List[nn.Module] = []
        enc_layers += [nn.Linear(cfg.in_dim, H)]
        if cfg.layernorm: enc_layers += [nn.LayerNorm(H)]
        enc_layers += [act, nn.Dropout(cfg.dropout)]
        for _ in range(cfg.depth - 1):
            enc_layers += [nn.Linear(H, H)]
            if cfg.layernorm: enc_layers += [nn.LayerNorm(H)]
            enc_layers += [act, nn.Dropout(cfg.dropout)]
        enc_layers += [nn.Linear(H, cfg.bottleneck)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        dec_layers += [nn.Linear(cfg.bottleneck, H)]
        if cfg.layernorm: dec_layers += [nn.LayerNorm(H)]
        dec_layers += [act, nn.Dropout(cfg.dropout)]
        for _ in range(cfg.depth - 1):
            dec_layers += [nn.Linear(H, H)]
            if cfg.layernorm: dec_layers += [nn.LayerNorm(H)]
            dec_layers += [act, nn.Dropout(cfg.dropout)]
        dec_layers += [nn.Linear(H, cfg.in_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    # -------- public API --------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., D_any]  -> pad/trim to in_dim -> encode -> [..., bottleneck]
        """
        x2d, leading, _ = _flatten_last(x)
        x2d = _align_last_dim(x2d, self.cfg.in_dim)
        z2d = self.encoder(x2d)
        return _unflatten_last(z2d, leading)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [..., Z_any]  -> pad/trim to bottleneck -> decode -> [..., in_dim]
        """
        z2d, leading, _ = _flatten_last(z)
        z2d = _align_last_dim(z2d, self.cfg.bottleneck)
        y2d = self.decoder(z2d)
        return _unflatten_last(y2d, leading)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autoencoder pass: [..., D_any] -> (align to in_dim) -> encode -> decode -> [..., in_dim]
        """
        x2d, leading, _ = _flatten_last(x)
        x2d = _align_last_dim(x2d, self.cfg.in_dim)
        z2d = self.encoder(x2d)
        y2d = self.decoder(z2d)
        return _unflatten_last(y2d, leading)

    # -------- I/O helpers --------
    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg.__dict__}, path)

    @staticmethod
    def load(path: str, map_location: Optional[str | torch.device] = None) -> "FcAutoEncoder":
        chk = torch.load(path, map_location=map_location)
        cfg = FcAeConfig(**chk["cfg"])
        model = FcAutoEncoder(cfg)
        model.load_state_dict(chk["state_dict"])
        return model


def build_fc_autoencoder(in_dim: int,
                         bottleneck: int = 256,
                         hidden_mult: float = 2.0,
                         depth: int = 2,
                         dropout: float = 0.0,
                         activation: str = "gelu",
                         layernorm: bool = True) -> FcAutoEncoder:
    return FcAutoEncoder(FcAeConfig(
        in_dim=in_dim,
        bottleneck=bottleneck,
        hidden_mult=hidden_mult,
        depth=depth,
        dropout=dropout,
        activation=activation,
        layernorm=layernorm,
    ))
