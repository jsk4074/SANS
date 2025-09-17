# scripts/train_fc_ae.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, glob, math, argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sans import build_model
from sans.ascent_ckpt import encode_audio_cond
from sans.fc_autoencoder import build_fc_autoencoder, FcAutoEncoder


class CondDataset(Dataset):
    """
    Builds a dataset of latent condition vectors by running encode_audio_cond on WAV files.
    Supports per-frame latents [T,D] or pooled latents [D].
    """
    def __init__(self, wav_glob: str, sr: int, ckpt: str, device: str):
        super().__init__()
        self.files = sorted(glob.glob(wav_glob))
        self.sr = int(sr)
        self.ldm = build_model(ckpt).to(device)
        self.device = device

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import librosa
        f = self.files[idx]
        wav, _ = librosa.load(f, sr=self.sr)
        x = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,T]
        cond = encode_audio_cond(self.ldm, x, sr=self.sr)  # [1,D] or [1,T,D]
        if cond.ndim == 3:
            cond = cond.squeeze(0)  # [T,D]
        else:
            cond = cond.squeeze(0)  # [D]
        return cond.detach()


def collate_pad(batch: list[torch.Tensor]) -> torch.Tensor:
    vecs = []
    for x in batch:
        vecs.append(x)

    return torch.cat(vecs, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Train fully-connected AE on condition vectors")
    ap.add_argument("--ckpt", type=str, required=True, help="AudioLDM ckpt for encoding")
    ap.add_argument("--wav_glob", type=str, required=True, help="Glob of training WAVs (e.g., data/train/*.wav)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=2048, help="Effective vectors per batch after flattening")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--hidden_mult", type=float, default=2.0)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--no_layernorm", action="store_true")
    ap.add_argument("--save", type=str, default="fc_ae.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # one probe to discover in_dim
    ds_probe = CondDataset(args.wav_glob, args.sr, args.ckpt, device)
    x0 = ds_probe[0]
    in_dim = x0.shape[-1]

    model = build_fc_autoencoder(
        in_dim=in_dim,
        bottleneck=args.bottleneck,
        hidden_mult=args.hidden_mult,
        depth=args.depth,
        dropout=args.dropout,
        activation=args.activation,
        layernorm=not args.no_layernorm,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, maximize=True)
    loss_mse = nn.MSELoss()
    dl = DataLoader(
        ds_probe, 
        batch_size=1, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=lambda b: collate_pad(b)
    )

    model.train()
    for ep in range(1, args.epochs + 1):
        total = 0.0
        count = 0
        for vecs in dl:  # vecs: [N,D]
            vecs = vecs.to(device)
            recon = model(vecs)
            loss = loss_mse(recon, vecs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * vecs.size(0)
            count += vecs.size(0)
        print(f"[epoch {ep}] loss={total / max(count,1):.6f}")

    model.eval()
    model.save(args.save)
    print(f"saved FCAE: {args.save} (in_dim={in_dim})")


if __name__ == "__main__":
    main()
