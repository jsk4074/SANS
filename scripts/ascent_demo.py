# sans/ascent.py
import torch
import torch.nn as nn
from torch import optim
from typing import Callable, Optional, Dict, Any

def _project_l2(e: torch.Tensor, e0: torch.Tensor, tau: float) -> torch.Tensor:
    d = e - e0
    n = d.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    over = (n > tau).float()
    return (e0 + d * (tau / n)) * over + e * (1 - over)

def _seed(seed: int, device: torch.device):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g

class DiffusersAudioLDMWrapper:
    """
    Minimal wrapper around a Diffusers AudioLDM pipeline so we can:
      1) get prompt embeddings; 2) generate with prompt_embeds and gradients.
    """
    def __init__(self, pipe):
        self.pipe = pipe
        self.device = pipe.device

    @torch.no_grad()
    def encode_prompt(self, text: str) -> torch.Tensor:
        tk = self.pipe.tokenizer(
            text,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tk = {k: v.to(self.device) for k, v in tk.items()}
        return self.pipe.text_encoder(**tk)[0]  # [B=1, T, D]

    def generate_from_embeds(
        self,
        prompt_embeds: torch.Tensor,
        num_inference_steps: int = 12,
        guidance_scale: float = 2.5,
        eta: float = 0.0,
        audio_length_in_s: float = 5.0,
        seed: int = 1234,
        extra: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        kw = dict(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,  # DDIM; keep 0.0 for deterministic & autograd-friendly
            audio_length_in_s=audio_length_in_s,
            output_type="pt",                      # keep gradients
            generator=_seed(seed, self.device),
        )
        if extra:
            kw.update(extra)
        out = self.pipe(prompt_embeds=prompt_embeds, **kw)
        wave = out.audios if hasattr(out, "audios") else out
        if isinstance(wave, list):
            wave = torch.stack(wave, dim=0)
        return wave  # [B, T]

def optimize_prompt_embeds(
    pipew: DiffusersAudioLDMWrapper,
    init_embeds: torch.Tensor,                        # [B, T, D] or [1, T, D]
    objective_fn: Callable[[torch.Tensor], torch.Tensor],  # waveform -> scalar
    steps: int = 20,
    inner_steps: int = 12,
    lr: float = 1e-2,
    l2_anchor: float = 1e-3,
    tau: float = 2.0,
    guidance_scale: float = 2.5,
    audio_length_in_s: float = 5.0,
    seed: int = 1234,
    detach_every: int = 0,
    extra: Optional[Dict[str, Any]] = None,
):
    e0 = init_embeds.detach().clone()
    e  = nn.Parameter(init_embeds.clone().requires_grad_(True))
    opt = optim.Adam([e], lr=lr)
    last_audio = None

    for t in range(steps):
        opt.zero_grad(set_to_none=True)

        audio = pipew.generate_from_embeds(
            e,
            num_inference_steps=inner_steps,
            guidance_scale=guidance_scale,
            audio_length_in_s=audio_length_in_s,
            seed=seed,
            extra=extra,
        )
        last_audio = audio
        score = objective_fn(audio)
        if score.ndim > 0:
            score = score.mean()

        reg = l2_anchor * (e - e0).pow(2).mean()
        loss = -(score) + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_([e], 5.0)
        opt.step()

        with torch.no_grad():
            e.data = _project_l2(e.data, e0, tau)
        if detach_every and ((t + 1) % detach_every == 0):
            e.data = e.data.detach().clone().requires_grad_(True)

    return e.detach(), last_audio
