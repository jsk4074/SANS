# sans/objectives.py
import torch
import torch.nn.functional as F
from typing import Callable

def ae_recon_error_objective(ae_model, to_model_input: Callable[[torch.Tensor], torch.Tensor]):
    def _obj(audio_wave_pt: torch.Tensor) -> torch.Tensor:
        x = to_model_input(audio_wave_pt)  
        x_hat = ae_model(x)
        return F.l1_loss(x_hat, x, reduction="none").mean(dim=list(range(1, x.ndim))).mean()
    return _obj

def nn_distance_objective(encoder, normal_bank: torch.Tensor):
    def _obj(audio_wave_pt: torch.Tensor) -> torch.Tensor:
        z = encoder(audio_wave_pt)  # [B, D]
        d2 = (z.pow(2).sum(-1, keepdim=True)
              + normal_bank.pow(2).sum(-1)[None, :]
              - 2 * z @ normal_bank.T)
        k = min(5, normal_bank.size(0))
        return d2.topk(k, dim=1, largest=True).values.mean()
    return _obj

# Simple autoencoder objectives 
def make_ae_recon_objective(ae_model, to_mel_fn):
    import torch.nn.functional as F
    def _obj(wave):
        mel = to_mel_fn(wave)
        recon = ae_model(mel)
        return F.l1_loss(recon, mel, reduction="none").mean()
    return _obj
