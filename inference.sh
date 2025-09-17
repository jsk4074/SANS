# python sample.py \
#   --ckpt /home/tori/.cache/audioldm/audioldm-s-full.ckpt \
#   --ref /home/tori/workspace/eval_2025/AutoTrash/test/section_00_0000.wav \
#   --ascent_steps 25 \
#   --inner_steps 12 \
#   --fc_ae fc_ae.pt \
#   --fc_ae_out recon

# python sample_combined.py \
#   --ckpt /home/tori/.cache/audioldm/audioldm-s-full.ckpt \
#   --ref /home/tori/workspace/eval_2025/AutoTrash/test/section_00_0000.wav \
#   --ascent_steps 25 --inner_steps 12 --guidance 2.5 \
#   --fc_ae fc_ae.pt --fc_ae_out recon --fcae_in_loop

# python sample_combined_effect.py \
#   --ckpt /home/tori/.cache/audioldm/audioldm-s-full.ckpt \
#   --ref /home/tori/workspace/eval_2025/AutoTrash/test/section_00_0000.wav \
#   --probe_cond \

python sample_combined_effect.py \
  --ckpt /home/tori/.cache/audioldm/audioldm-s-full.ckpt \
  --ref ../eval_2025/AutoTrash/test/section_00_0000.wav \
  --ascent_steps 25 --inner_steps 12 \
  --fc_ae fc_ae.pt --fc_ae_out recon --fcae_in_loop \
  --fcae_alpha 3.0 --fcae_dropout_eval --probe_cond
