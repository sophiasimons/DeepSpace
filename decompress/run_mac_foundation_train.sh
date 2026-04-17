#!/bin/bash
# ---------------------------------------------------------------------------
# Foundation model training — Mac / Apple Silicon (MPS)
#
# This trains a single model on ALL region datasets placed under
# data/multi_region/.  Each sub-folder must follow the standard layout:
#
#   data/multi_region/
#       region_a_16_256/
#           hr_256/
#           sr_16_256/
#       region_b_16_256/
#           hr_256/
#           sr_16_256/
#       ...
#
# After training completes, copy the checkpoint to use with finetune_srwddgan.py
# or compare it directly via benchmark/compare_models.py.
# ---------------------------------------------------------------------------

set -e
cd "$(dirname "$0")"

python train_srwddgan.py \
    --dataset foundation_multi_region \
    --datadir ./data/multi_region \
    --data_len_per_region 500 \
    --exp foundation_model \
    --image_size 256 \
    --current_resolution 256 \
    --num_channels 12 \
    --num_channels_dae 128 \
    --ch_mult 1 1 2 2 4 4 \
    --num_res_blocks 2 \
    --attn_resolutions 16 \
    --use_pytorch_wavelet \
    --rec_loss \
    --batch_size 2 \
    --num_epoch 10 \
    --lr_g 1.5e-4 \
    --lr_d 1e-4 \
    --num_timesteps 4 \
    --save_ckpt_every 100 \
    --save_content_every 100 \
    --l_resolution 16 \
    --h_resolution 256 \
    --num_workers 1 \
    --master_port 6003

echo ""
echo "Foundation model training complete."
echo "Checkpoint location: content/foundation_multi_region/foundation_model/"
