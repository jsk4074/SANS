import torch
import torch.nn as nn
from torch import optim
from typing import Callable, Dict, Any, Optional

def _project_l2(e, e0, tau):
    d = e - e0
    n = d.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    mask = (n > tau).float()
    return (e0 + d * (tau / n)) * mask + e * (1 - mask)

@torch.no_grad()
def _seed(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g

def get_audio_cond(ldm, wav_tensor, sr: int):
    
    mel = ldm.audio_augmentations(wav_tensor, sr)          
    cond = ldm.audio_cond_encoder(mel)                     
    return cond  

def generate_from_audio_cond(
    ldm,
    cond_emb: torch.Tensor,
    *,
    steps: int = 12,
    cfg: float = 2.5,
    duration_s: float = 5.0,
    seed: int = 1234,
    extra: Optional[Dict[str, Any]] = None,
):
    g = _seed(seed, cond_emb.device)
    kwargs = dict(steps=steps, guidance_scale=cfg, duration=duration_s, generator=g)
    if extra: kwargs.update(extra)
    
    out = ldm.generate_with_audio_cond(cond_emb=cond_emb, **kwargs)
    
    return out  

def optimize_audio_cond(
    ldm,
    init_cond: torch.Tensor,
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    outer_steps: int = 25,
    inner_steps: int = 12,
    lr: float = 1e-2,
    l2_anchor: float = 1e-3,
    tau: float = 2.0,
    cfg: float = 2.5,
    duration_s: float = 5.0,
    seed: int = 1234,
):
    e0 = init_cond.detach().clone()
    e  = nn.Parameter(init_cond.clone().requires_grad_(True))
    opt = optim.Adam([e], lr=lr)
    last = None
    for _ in range(outer_steps):
        opt.zero_grad(set_to_none=True)
        audio = generate_from_audio_cond(
            ldm, e, steps=inner_steps, cfg=cfg, duration_s=duration_s, seed=seed
        )
        last = audio
        score = objective_fn(audio)
        if score.ndim: score = score.mean()
        loss = -(score) + l2_anchor * (e - e0).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([e], 5.0)
        opt.step()
        with torch.no_grad():
            e.data = _project_l2(e.data, e0, tau)
    return e.detach(), last
