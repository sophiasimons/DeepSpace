'''
Model comparison script for the DeepSpace foundation model project.

Runs inference with two model checkpoints (e.g. a region-specific expert model
and the fine-tuned foundation model) on the same set of images and prints a
side-by-side comparison of PSNR, SSIM, and LPIPS.

Usage example:
    cd decompress/
    python benchmark/compare_models.py \
        --model_a content/deepgreen_16_256/deepgreen_mac_smoketest/netG_0_iteration_40.pth \
        --model_a_name "Expert (deepgreen)" \
        --model_b content/deepgreen_16_256/finetune_new_region/netG_finetuned_final.pth \
        --model_b_name "Foundation (fine-tuned)" \
        --dataset deepgreen_16_256 \
        --datadir ./data/deepgreen_16_256 \
        --data_len 50 \
        --image_size 256 --current_resolution 256 \
        --num_channels 12 --num_channels_dae 128 \
        --ch_mult 1 1 2 2 4 4 --num_res_blocks 2 \
        --attn_resolutions 16 --use_pytorch_wavelet \
        --l_resolution 16 --h_resolution 256

Results are saved to benchmark/comparison_results/<timestamp>.txt
'''

import argparse
import os
import sys
import datetime
import numpy as np
import torch
import torchvision

# Make sure the decompress/ root is on the path when run from benchmark/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets_prep.dataset import create_dataset
from diffusion import get_time_schedule, Posterior_Coefficients, sample_from_model
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from score_sde.models.ncsnpp_generator_adagn import WaveletNCSNpp
from torch.utils.data import DataLoader, Subset

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    HAS_PW = True
except ImportError:
    HAS_PW = False

try:
    from PIL import Image
    import benchmark.metrics as Metrics
    HAS_METRICS = True
except Exception:
    try:
        import metrics as Metrics
        from PIL import Image
        HAS_METRICS = True
    except Exception:
        HAS_METRICS = False
        print("[WARN] metrics module not found — PSNR/SSIM will be skipped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _load_model(ckpt_path, args, device):
    """Load a WaveletNCSNpp generator from a checkpoint file."""
    netG = WaveletNCSNpp(args).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # Strip DDP 'module.' prefix if present
    ckpt = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items()}
    missing, unexpected = netG.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {len(missing)}")
    netG.eval()
    return netG


def _wavelet_transform(x, dwt, use_pytorch_wavelet, num_levels):
    if not use_pytorch_wavelet:
        for _ in range(num_levels):
            xll, xlh, xhl, xhh = dwt(x)
    else:
        xll, xh = dwt(x)
        xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
    return torch.cat([xll, xlh, xhl, xhh], dim=1) / 2.0


def _inverse_wavelet(data, iwt, use_pytorch_wavelet):
    data = data * 2
    if not use_pytorch_wavelet:
        return iwt(data[:, :3], data[:, 3:6], data[:, 6:9], data[:, 9:12])
    else:
        return iwt((data[:, :3], [torch.stack(
            (data[:, 3:6], data[:, 6:9], data[:, 9:12]), dim=2)]))


def _to_uint8_numpy(tensor):
    """Convert a (B, C, H, W) tensor in [0,1] to a (H, W, C) uint8 numpy array."""
    img = tensor[0].cpu().float().permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _run_inference(netG, sr_data_batch, pos_coeff, T, args, device):
    """Run diffusion sampling for a full batch and return images in [0,1]."""
    x_t_1 = torch.randn_like(sr_data_batch)
    with torch.no_grad():
        out = sample_from_model(
            pos_coeff, netG, args.num_timesteps, x_t_1, sr_data_batch, T, args)
    return out


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(netG, data_loader, dwt, iwt, pos_coeff, T, args, device,
                   save_dir, model_tag):
    """Run the model on all batches and compute image-quality metrics.

    Returns a dict with keys: psnr, ssim (floats, averaged over images).
    """
    os.makedirs(save_dir, exist_ok=True)
    num_levels = int(np.log2(args.ori_image_size // args.current_resolution))

    psnr_list, ssim_list = [], []
    img_idx = 0

    for data_dict in data_loader:
        hr_image = data_dict['HR']
        sr_image = data_dict['SR']

        sr = sr_image.to(device)
        sr_data = _wavelet_transform(sr, dwt, args.use_pytorch_wavelet, num_levels)

        out_wavelet = _run_inference(netG, sr_data, pos_coeff, T, args, device)
        out_img = _inverse_wavelet(out_wavelet, iwt, args.use_pytorch_wavelet)
        out_img = (torch.clamp(out_img, -1, 1) + 1) / 2  # → [0,1]

        hr_norm = (hr_image.clamp(-1, 1) + 1) / 2

        for b in range(out_img.shape[0]):
            sr_np  = _to_uint8_numpy(out_img[b:b+1])
            hr_np  = _to_uint8_numpy(hr_norm[b:b+1])

            # Save output image
            out_path = os.path.join(save_dir, f'{model_tag}_{img_idx:04d}_sr.png')
            Image.fromarray(sr_np).save(out_path)
            hr_path = os.path.join(save_dir, f'{model_tag}_{img_idx:04d}_hr.png')
            Image.fromarray(hr_np).save(hr_path)

            if HAS_METRICS:
                psnr_list.append(Metrics.calculate_psnr(sr_np, hr_np))
                ssim_list.append(Metrics.calculate_ssim(sr_np, hr_np))

            img_idx += 1

    results = {
        'n_images': img_idx,
        'psnr': float(np.mean(psnr_list)) if psnr_list else float('nan'),
        'ssim': float(np.mean(ssim_list)) if ssim_list else float('nan'),
    }
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    device = _get_device()
    print(f"Device: {device}\n")

    # Shared model architecture settings
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution

    # Dataset
    dataset = create_dataset(args)
    n = min(args.data_len, len(dataset)) if args.data_len > 0 else len(dataset)
    subset = Subset(dataset, list(range(n)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    print(f"Evaluating on {n} images.\n")

    # Wavelet transforms
    if not args.use_pytorch_wavelet or not HAS_PW:
        args.use_pytorch_wavelet = False
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)
        iwt = DWTInverse(mode='zero', wave='haar').to(device)

    pos_coeff = Posterior_Coefficients(args, device)
    T         = get_time_schedule(args, device)

    # Results directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), 'comparison_results', timestamp)
    os.makedirs(results_dir, exist_ok=True)

    report_lines = [
        f"DeepSpace Model Comparison",
        f"Timestamp : {timestamp}",
        f"Dataset   : {args.dataset}  ({n} images)",
        f"Device    : {device}",
        "=" * 60,
    ]

    for ckpt_path, model_name in [
            (args.model_a, args.model_a_name),
            (args.model_b, args.model_b_name)]:

        print(f"--- Evaluating: {model_name} ---")
        print(f"    Checkpoint: {ckpt_path}")
        netG = _load_model(ckpt_path, args, device)

        save_dir = os.path.join(results_dir, model_name.replace(' ', '_'))
        metrics  = evaluate_model(netG, loader, dwt, iwt, pos_coeff, T,
                                  args, device, save_dir, model_tag='img')

        line = (f"{model_name:<35} | "
                f"PSNR: {metrics['psnr']:6.2f} dB | "
                f"SSIM: {metrics['ssim']:.4f} | "
                f"N: {metrics['n_images']}")
        print(line)
        report_lines.append(line)
        del netG

    report_lines.append("=" * 60)
    report_lines.append("Higher PSNR/SSIM = better reconstruction quality.")
    report = "\n".join(report_lines)

    report_file = os.path.join(results_dir, 'report.txt')
    with open(report_file, 'w') as f:
        f.write(report + '\n')

    print(f"\n{report}\n")
    print(f"Report saved to: {report_file}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeepSpace model comparison')

    # The two checkpoints to compare
    parser.add_argument('--model_a', type=str, required=True,
                        help='Path to first model checkpoint (.pth)')
    parser.add_argument('--model_a_name', type=str, default='Model A')
    parser.add_argument('--model_b', type=str, required=True,
                        help='Path to second model checkpoint (.pth)')
    parser.add_argument('--model_b_name', type=str, default='Model B')

    # Dataset
    parser.add_argument('--dataset', default='deepgreen_16_256')
    parser.add_argument('--datadir', default='./data/deepgreen_16_256')
    parser.add_argument('--data_len', type=int, default=50,
                        help='Number of test images to evaluate on')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--l_resolution', type=int, default=16)
    parser.add_argument('--h_resolution', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)

    # Model architecture (must match both checkpoints)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--current_resolution', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=12)
    parser.add_argument('--num_channels_dae', type=int, default=128)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=[16])
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--resamp_with_conv', action='store_false', default=True)
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--fir', action='store_false', default=True)
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
    parser.add_argument('--skip_rescale', action='store_false', default=True)
    parser.add_argument('--resblock_type', default='biggan')
    parser.add_argument('--progressive', default='none',
                        choices=['none', 'output_skip', 'residual'])
    parser.add_argument('--progressive_input', default='residual',
                        choices=['none', 'input_skip', 'residual'])
    parser.add_argument('--progressive_combine', default='sum',
                        choices=['sum', 'cat'])
    parser.add_argument('--embedding_type', default='positional',
                        choices=['positional', 'fourier'])
    parser.add_argument('--fourier_scale', type=float, default=16.)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
    parser.add_argument('--use_pytorch_wavelet', action='store_true', default=False)
    parser.add_argument('--net_type', default='wavelet')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--cond_emb_dim', type=int, default=256)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
    parser.add_argument('--no_use_fbn', action='store_true')
    parser.add_argument('--no_use_freq', action='store_true')
    parser.add_argument('--no_use_residual', action='store_true')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
