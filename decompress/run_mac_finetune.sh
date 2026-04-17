#!/bin/bash
# ---------------------------------------------------------------------------
# Few-shot fine-tuning of the foundation model to a new region — Mac / MPS
#
# Prerequisites:
#   1. A trained foundation model checkpoint, e.g.:
#      content/foundation_multi_region/foundation_model/netG_5_iteration_200.pth
#
#   2. A new-region dataset with 50-100 images in the standard layout:
#      data/new_region_16_256/
#          hr_256/      ← high-res ground truth images
#          sr_16_256/   ← bicubic-upsampled low-res images (same filenames)
#
# Edit the two variables below, then run:
#   bash run_mac_finetune.sh
# ---------------------------------------------------------------------------

set -e
cd "$(dirname "$0")"

# ---- CONFIGURE THESE -------------------------------------------------------
FOUNDATION_CKPT="content/foundation_multi_region/foundation_model/netG_finetuned_final.pth"
NEW_REGION_DATADIR="./data/new_region_16_256"
NEW_REGION_NAME="new_region"   # used for the experiment folder name
# ----------------------------------------------------------------------------

python finetune_srwddgan.py \
    --foundation_ckpt "${FOUNDATION_CKPT}" \
    --dataset deepgreen_16_256 \
    --datadir "${NEW_REGION_DATADIR}" \
    --data_len 100 \
    --exp "finetune_${NEW_REGION_NAME}" \
    --num_epoch 20 \
    --batch_size 2 \
    --lr_g 5e-5 \
    --lr_d 2e-5 \
    --num_timesteps 4 \
    --save_ckpt_every 50 \
    --save_content_every 50 \
    --image_size 256 \
    --current_resolution 256 \
    --num_channels 12 \
    --num_channels_dae 128 \
    --ch_mult 1 1 2 2 4 4 \
    --num_res_blocks 2 \
    --attn_resolutions 16 \
    --use_pytorch_wavelet \
    --rec_loss \
    --l_resolution 16 \
    --h_resolution 256 \
    --num_workers 1

echo ""
echo "Fine-tuning complete."
echo "Checkpoint: content/deepgreen_16_256/finetune_${NEW_REGION_NAME}/netG_finetuned_final.pth"
echo ""
echo "To compare against an expert model, run:"
echo "  python benchmark/compare_models.py \\"
echo "    --model_a <expert_ckpt.pth> --model_a_name 'Expert' \\"
echo "    --model_b content/deepgreen_16_256/finetune_${NEW_REGION_NAME}/netG_finetuned_final.pth \\"
echo "    --model_b_name 'Foundation (fine-tuned)' \\"
echo "    --dataset deepgreen_16_256 --datadir ${NEW_REGION_DATADIR} \\"
echo "    --data_len 50 --use_pytorch_wavelet"
