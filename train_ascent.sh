python scripts/train_fc_ae.py \
    --ckpt /home/tori/.cache/audioldm/audioldm-s-full.ckpt \
    --wav_glob "../eval_2025/AutoTrash/test/*.wav" \
    --sr 16000 \
    --epochs 20 \
    --bottleneck 256 \
    --save fc_ae.pt \