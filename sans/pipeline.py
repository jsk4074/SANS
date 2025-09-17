import os

import argparse
import yaml
import torch
from torch import autocast
from tqdm import tqdm, trange

from sans import LatentDiffusion, seed_everything
from sans.utils import default_audioldm_config, get_duration, get_bit_depth, get_metadata, download_checkpoint
from sans.audio import wav_to_fbank, TacotronSTFT, read_wav_file
from sans.latent_diffusion.ddim import DDIMSampler
from einops import repeat
import contextlib
import os

def make_batch_for_text_to_audio(text, waveform=None, fbank=None, batchsize=1):
    text = [text] * batchsize
    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")
    
    if(fbank is None):
        fbank = torch.zeros((batchsize, 1024, 64))  # Not used, here to keep the code format
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize
        
    stft = torch.zeros((batchsize, 1024, 512))  # Not used

    if(waveform is None):
        waveform = torch.zeros((batchsize, 160000))  # Not used
    else:
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batchsize, -1)
        assert waveform.size(0) == batchsize
        
    fname = [""] * batchsize  # Not used
    
    batch = (
        fbank,
        stft,
        None,
        fname,
        waveform,
        text,
    )
    return batch

def round_up_duration(duration):
    return int(round(duration/2.5) + 1) * 2.5

def build_model(
    ckpt_path=None,
    config=None,
    model_name="audioldm-s-full"
):
    print("Load AudioLDM: %s", model_name)
    
    if(ckpt_path is None):
        ckpt_path = get_metadata()[model_name]["path"]
    
    if(not os.path.exists(ckpt_path)):
        download_checkpoint(model_name)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config(model_name)

    # Use text as condition instead of using waveform during training
    config["model"]["params"]["device"] = device
    config["model"]["params"]["cond_stage_key"] = "text"

    # No normalization here
    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    resume_from_checkpoint = ckpt_path

    checkpoint = torch.load(resume_from_checkpoint, map_location=device)
    '''Original. Here is a bug that, an unexpected key "cond_stage_model.model.text_branch.embeddings.position_ids" exists in the checkpoint file. '''
    # latent_diffusion.load_state_dict(checkpoint["state_dict"])
    '''2023.10.17 Fix the bug by setting the paramer "strict" as "False" to ignore the unexpected key. '''
    latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    latent_diffusion.cond_stage_model.embed_mode = "text"
    return latent_diffusion

def duration_to_latent_t_size(duration):
    return int(duration * 25.6)

def set_cond_audio(latent_diffusion):
    latent_diffusion.cond_stage_key = "waveform"
    latent_diffusion.cond_stage_model.embed_mode="audio"
    return latent_diffusion

def set_cond_text(latent_diffusion):
    latent_diffusion.cond_stage_key = "text"
    latent_diffusion.cond_stage_model.embed_mode="text"
    return latent_diffusion

def style_transfer(
    latent_diffusion,
    text,
    original_audio_file_path,
    transfer_strength,
    seed=42,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    ddim_steps=200,
    config=None,
    output_type=None,
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    assert original_audio_file_path is not None, "You need to provide the original audio file path"
    
    audio_file_duration = get_duration(original_audio_file_path)
    
    assert get_bit_depth(original_audio_file_path) == 16, "The bit depth of the original audio file %s must be 16" % original_audio_file_path
    
    # if(duration > 20):
    #     print("Warning: The duration of the audio file %s must be less than 20 seconds. Longer duration will result in Nan in model output (we are still debugging that); Automatically set duration to 20 seconds")
    #     duration = 20
    
    if(duration > audio_file_duration):
        print("Warning: Duration you specified %s-seconds must equal or smaller than the audio file duration %ss" % (duration, audio_file_duration))
        duration = round_up_duration(audio_file_duration)
        print("Set new duration as %s-seconds" % duration)

    # duration = round_up_duration(duration)
    
    latent_diffusion = set_cond_text(latent_diffusion)

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config()

    seed_everything(int(seed))
    # latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    latent_diffusion.cond_stage_model.embed_mode = "text"

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    mel, _, _ = wav_to_fbank(
        original_audio_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT
    )
    mel = mel.unsqueeze(0).unsqueeze(0).to(device)
    mel = repeat(mel, "1 ... -> b ...", b=batchsize)
    init_latent = latent_diffusion.get_first_stage_encoding(
        latent_diffusion.encode_first_stage(mel)
    )  # move to latent space, encode and sample
    if(torch.max(torch.abs(init_latent)) > 1e2):
        init_latent = torch.clip(init_latent, min=-10, max=10)
    sampler = DDIMSampler(latent_diffusion)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=1.0, verbose=False)

    t_enc = int(transfer_strength * ddim_steps)
    prompts = text

    amp_ctx = autocast("cuda") if device.type == "cuda" else contextlib.nullcontext()
    with torch.no_grad():
        with amp_ctx:
            with latent_diffusion.ema_scope():
                uc = None
                if guidance_scale != 1.0:
                    uc = latent_diffusion.cond_stage_model.get_unconditional_condition(
                        batchsize
                    )

                c = latent_diffusion.get_learned_conditioning([prompts] * batchsize)
                z_enc = sampler.stochastic_encode(
                    init_latent, torch.tensor([t_enc] * batchsize).to(device)
                )
                samples = sampler.decode(
                    z_enc,
                    c,
                    t_enc,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc,
                )
                # x_samples = latent_diffusion.decode_first_stage(samples) # Will result in Nan in output
                # print(torch.sum(torch.isnan(samples)))
                x_samples = latent_diffusion.decode_first_stage(samples)
                # print(x_samples)
                x_samples = latent_diffusion.decode_first_stage(samples[:,:,:-3,:])
                # print(x_samples)
                waveform = latent_diffusion.first_stage_model.decode_to_waveform(
                    x_samples
                )
    return mel
    # return waveform

# --- New: build a Diffusers AudioLDM pipeline ---
def build_diffusers(repo_id: str = "cvssp/audioldm-s-full-v2",
                    device: str = None,
                    dtype=None):
    import torch
    from diffusers import AudioLDMPipeline
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=dtype)
    return pipe.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))

def ascent_transfer(
    ldm_or_pipe,            
    text: str,
    *,
    objective_fn,           
    ascent_steps: int = 25,
    inner_ddim_steps: int = 12,
    guidance_scale: float = 2.5,
    duration: float = 5.0,
    tau: float = 2.0,
    lr: float = 1e-2,
    seed: int = 1234,
    output_type: str = "mel",
    to_mel=None,            
):
    import torch
    from .ascent import DiffusersAudioLDMWrapper, optimize_prompt_embeds

    assert hasattr(ldm_or_pipe, "tokenizer"), \
        "ascent_transfer requires a Diffusers AudioLDM pipeline. Use build_diffusers(...)."

    pipew = DiffusersAudioLDMWrapper(ldm_or_pipe)
    with torch.no_grad():
        e0 = pipew.encode_prompt(text or "")

    e_star, wave = optimize_prompt_embeds(
        pipew,
        init_embeds=e0,
        objective_fn=objective_fn,
        steps=ascent_steps,
        inner_steps=inner_ddim_steps,
        lr=lr,
        l2_anchor=1e-3,
        tau=tau,
        guidance_scale=guidance_scale,
        audio_length_in_s=duration,
        seed=seed,
    )

    if output_type == "waveform":
        return wave
    if to_mel is not None:
        return to_mel(wave)
    try:
        from .audio_utils import WaveToMel
        return WaveToMel()(wave.to(wave.device))
    except Exception:
        return wave
