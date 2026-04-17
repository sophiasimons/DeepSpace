#!/bin/sh
# Mac smoke-test: trains for a tiny number of iterations just to verify the
# full pipeline (train → checkpoint → test) runs without errors.
# Output quality will be poor — this is only for pipeline validation.
#
# Expected runtime: ~10-30 min on Apple Silicon MPS, ~1-2 hr on CPU
#
# Usage:  bash run_mac_smoketest.sh

export PYTHONPATH=$(pwd):$PYTHONPATH

EXP_NAME="deepgreen_mac_smoketest"

echo "==> Starting Mac smoke-test training (deepgreen_16_256)"
python train_srwddgan.py \
    --dataset deepgreen_16_256 \
    --image_size 256 \
    --exp $EXP_NAME \
    --num_channels 24 \
    --num_channels_dae 64 \
    --ch_mult 1 2 2 2 4 \
    --num_timesteps 2 \
    --num_res_blocks 2 \
    --batch_size 4 \
    --num_epoch 1 \
    --ngf 64 \
    --embedding_type positional \
    --use_ema \
    --r1_gamma 2. \
    --cond_emb_dim 256 \
    --lr_d 1e-4 \
    --lr_g 2e-4 \
    --lazy_reg 10 \
    --save_content \
    --save_ckpt_every 40 \
    --save_content_every 40 \
    --datadir ./data/deepgreen_16_256/ \
    --num_process_per_node 1 \
    --master_port 6002 \
    --current_resolution 128 \
    --attn_resolution 16 \
    --num_disc_layers 6 \
    --rec_loss \
    --l_resolution 16 \
    --h_resolution 256 \
    --use_pytorch_wavelet \
    --net_type wavelet \
    --data_len 200

echo ""
echo "==> Smoke-test training done."
echo "    Checkpoint saved to: content/deepgreen_16_256/${EXP_NAME}/"
echo ""
echo "==> Now running inference with the saved checkpoint..."
echo "    (Find the epoch/iteration numbers from the filename above)"
echo ""
echo "    Example test command:"
echo "    python test_srwddgan.py \\"
echo "      --dataset deepgreen_16_256 --image_size 256 \\"
echo "      --exp $EXP_NAME \\"
echo "      --num_channels 24 --num_channels_dae 64 \\"
echo "      --ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2 \\"
echo "      --epoch_id <EPOCH> --num_iters <ITERS> \\"
echo "      --current_resolution 128 --attn_resolutions 16 \\"
echo "      --net_type wavelet \\"
echo "      --l_resolution 16 --h_resolution 256 \\"
echo "      --datadir ./data/deepgreen_16_256/ \\"
echo "      --batch_size 4 \\"
echo "      --use_pytorch_wavelet"
