# sans/audio_utils.py
import torch
import torchaudio as ta

class WaveToMel(torch.nn.Module):
    def __init__(
        self,
        sr=16000, n_fft=1024, hop=160,
        n_mels=128, fmin=20, fmax=8000, power=2.0
    ):
        super().__init__()
        self.melspec = ta.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            f_min=fmin, f_max=fmax, n_mels=n_mels, center=True, power=power
        )
        self.amp2db = ta.transforms.AmplitudeToDB(stype="power")

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.ndim == 2:   
            wave = wave.unsqueeze(1)
        mel = self.melspec(wave)
        mel = self.amp2db(mel)     
        return mel
