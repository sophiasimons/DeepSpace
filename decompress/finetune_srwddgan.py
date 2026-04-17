'''
Few-shot fine-tuning script for the DeepSpace foundation model.

Given a pretrained foundation model checkpoint (netG_*.pth), this script
adapts it to a new region using only 50-100 images.  The generator backbone
is frozen; only the final upsampling / output layers are trained so that the
general diffusion prior is preserved while the model specialises to the new
region's statistics.

Typical usage (Mac / MPS):
    python finetune_srwddgan.py \
        --foundation_ckpt content/foundation/foundation_exp/netG_0_iteration_40.pth \
        --dataset deepgreen_16_256 \
        --datadir ./data/new_region_16_256 \
        --data_len 100 \
        --exp finetune_new_region \
        --num_epoch 20 \
        --batch_size 2 \
        --lr_g 5e-5 \
        --lr_d 2e-5 \
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
        --h_resolution 256 \
        --l_resolution 16
'''

import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from datasets_prep.dataset import create_dataset
from diffusion import (sample_from_model, sample_posterior, q_sample_pairs,
                       get_time_schedule, Posterior_Coefficients,
                       Diffusion_Coefficients)
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from utils import copy_source, broadcast_params
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _freeze_backbone(netG):
    """Freeze all generator layers except the final output/upsampling blocks.

    We keep the last ResBlock and output conv trainable so the model can adapt
    its high-frequency texture predictions to the new region, while the deep
    encoder/decoder weights that capture general image structure are frozen.
    """
    for name, param in netG.named_parameters():
        # Keep the last resolution block and output projection trainable.
        # Names ending with 'output_layer', 'up_*', or the top-level 'end'
        # module differ by architecture version; we match conservatively.
        keep = any(tag in name for tag in ['output_layer', 'end', 'up_4',
                                            'up_3', 'res_block_5',
                                            'res_block_4'])
        param.requires_grad = keep

    trainable = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    total = sum(p.numel() for p in netG.parameters())
    print(f"[Freeze] Generator: {trainable:,} / {total:,} parameters trainable "
          f"({100 * trainable / total:.1f}%)")


def grad_penalty_call(args, D_real, x_t):
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True
    )[0]
    grad_penalty = (
        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()
    grad_penalty = args.r1_gamma / 2 * grad_penalty
    grad_penalty.backward()


# ---------------------------------------------------------------------------
# Main fine-tune function
# ---------------------------------------------------------------------------

def finetune(args):
    from EMA import EMA
    from score_sde.models.discriminator import Discriminator_large
    from score_sde.models.ncsnpp_generator_adagn import WaveletNCSNpp

    torch.manual_seed(args.seed)
    device = _get_device()
    print(f"Using device: {device}")

    # ---- Output directory ----
    exp_path = os.path.join('content', args.dataset, args.exp)
    os.makedirs(exp_path, exist_ok=True)
    copy_source(__file__, exp_path)

    # ---- Dataset (few-shot: only args.data_len images) ----
    dataset = create_dataset(args)
    print(f"Fine-tune dataset size: {len(dataset)} images")

    # Sequential split: first 80 % for training, rest for quick eval
    train_size = max(1, int(0.8 * len(dataset)))
    test_size = len(dataset) - train_size
    indices = list(range(len(dataset)))
    train_set = Subset(dataset, indices[:train_size])
    test_set  = Subset(dataset, indices[train_size:]) if test_size > 0 else Subset(dataset, indices[:2])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True)

    test_loader = DataLoader(
        test_set, batch_size=min(args.batch_size, len(test_set)), shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available())

    test_samples   = next(iter(test_loader))
    test_hr_image  = test_samples['HR']
    test_sr_image  = test_samples['SR']
    torchvision.utils.save_image(
        test_hr_image, os.path.join(exp_path, 'test_hr.png'), normalize=True)
    torchvision.utils.save_image(
        test_sr_image, os.path.join(exp_path, 'test_lr.png'), normalize=True)

    # ---- Build models ----
    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution

    netG = WaveletNCSNpp(args).to(device)
    netD = Discriminator_large(
        nc=args.num_channels, ngf=args.ngf,
        t_emb_dim=args.t_emb_dim,
        act=nn.LeakyReLU(0.2),
        num_layers=args.num_disc_layers).to(device)

    # ---- Load foundation model checkpoint into the generator ----
    print(f"Loading foundation model from: {args.foundation_ckpt}")
    ckpt = torch.load(args.foundation_ckpt, map_location=device)
    # Strip 'module.' prefix that DDP adds
    ckpt = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items()}
    missing, unexpected = netG.load_state_dict(ckpt, strict=False)
    print(f"  Missing keys : {len(missing)}")
    print(f"  Unexpected   : {len(unexpected)}")

    # ---- Freeze backbone; only fine-tune last layers ----
    _freeze_backbone(netG)
    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    # ---- Optimisers (lower LR for fine-tuning) ----
    optimizerG = optim.Adam(
        filter(lambda p: p.requires_grad, netG.parameters()),
        lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(
        filter(lambda p: p.requires_grad, netD.parameters()),
        lr=args.lr_d, betas=(args.beta1, args.beta2))

    if args.use_ema:
        from EMA import EMA as EMAWrapper
        optimizerG = EMAWrapper(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerG, args.num_epoch, eta_min=1e-6)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerD, args.num_epoch, eta_min=1e-6)

    # ---- Wavelet transforms ----
    if not args.use_pytorch_wavelet:
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)
        iwt = DWTInverse(mode='zero', wave='haar').to(device)
    num_levels = int(np.log2(args.ori_image_size // args.current_resolution))

    # Pre-compute wavelet of the fixed test SR batch
    test_sr = test_sr_image.to(device, non_blocking=torch.cuda.is_available())
    if not args.use_pytorch_wavelet:
        for _ in range(num_levels):
            test_srll, test_srlh, test_srhl, test_srhh = dwt(test_sr)
    else:
        test_srll, test_srh = dwt(test_sr)
        test_srlh, test_srhl, test_srhh = torch.unbind(test_srh[0], dim=2)
    test_sr_data = torch.cat([test_srll, test_srlh, test_srhl, test_srhh], dim=1) / 2.0

    coeff     = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T         = get_time_schedule(args, device)

    global_step = 0

    # ---- Training loop ----
    print(f"\nStarting fine-tuning for {args.num_epoch} epoch(s) "
          f"on {train_size} images.\n")

    for epoch in range(args.num_epoch):
        netG.train()
        netD.train()

        for iteration, data_dict in enumerate(train_loader):
            hr_image = data_dict['HR']
            sr_image = data_dict['SR']

            x0 = hr_image.to(device, non_blocking=torch.cuda.is_available())

            # Wavelet-transform the HR batch
            if not args.use_pytorch_wavelet:
                for _ in range(num_levels):
                    xll, xlh, xhl, xhh = dwt(x0)
            else:
                xll, xh = dwt(x0)
                xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
            real_data = torch.cat([xll, xlh, xhl, xhh], dim=1) / 2.0

            sr = sr_image.to(device, non_blocking=torch.cuda.is_available())
            if not args.use_pytorch_wavelet:
                for _ in range(num_levels):
                    srll, srlh, srhl, srhh = dwt(sr)
            else:
                srll, srh = dwt(sr)
                srlh, srhl, srhh = torch.unbind(srh[0], dim=2)
            sr_data = torch.cat([srll, srlh, srhl, srhh], dim=1) / 2.0

            # ---- Discriminator update ----
            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()
            for p in netG.parameters():
                p.requires_grad = False

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            D_real   = netD(x_t, t, x_tp1.detach()).view(-1)
            errD_real = F.softplus(-D_real).mean()
            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None or global_step % args.lazy_reg == 0:
                grad_penalty_call(args, D_real, x_t)

            x_0_predict  = netG(x_tp1.detach(), t, sr_data)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            output       = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errD_fake    = F.softplus(output).mean()
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # ---- Generator update ----
            for p in netD.parameters():
                p.requires_grad = False
            # Re-enable only the trainable generator params
            for p in netG.parameters():
                p.requires_grad = p.requires_grad  # already set by _freeze_backbone
            netG.zero_grad()

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_0_predict  = netG(x_tp1.detach(), t, sr_data)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            output       = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errG         = F.softplus(-output).mean()

            if args.rec_loss:
                errG = errG + F.l1_loss(x_0_predict, real_data)

            errG.backward()
            optimizerG.step()

            global_step += 1

            if global_step % 10 == 0:
                print(f"Epoch {epoch} iter {global_step}  G: {errG.item():.4f}  D: {errD.item():.4f}")

            # Save checkpoint
            if global_step % args.save_ckpt_every == 0:
                torch.save({k: v.cpu() for k, v in netG.state_dict().items()},
                           os.path.join(exp_path,
                                        f'netG_{epoch}_iteration_{global_step}.pth'))
                print("  => checkpoint saved.")

            # Save sample images
            if global_step % args.save_content_every == 0:
                netG.eval()
                with torch.no_grad():
                    x_t_1    = torch.randn_like(real_data)
                    resoluted = sample_from_model(
                        pos_coeff, netG, args.num_timesteps, x_t_1, test_sr_data, T, args)
                    resoluted *= 2
                    if not args.use_pytorch_wavelet:
                        resoluted = iwt(resoluted[:, :3], resoluted[:, 3:6],
                                        resoluted[:, 6:9], resoluted[:, 9:12])
                    else:
                        resoluted = iwt((resoluted[:, :3], [torch.stack(
                            (resoluted[:, 3:6], resoluted[:, 6:9],
                             resoluted[:, 9:12]), dim=2)]))
                    resoluted = (torch.clamp(resoluted, -1, 1) + 1) / 2
                    torchvision.utils.save_image(
                        resoluted,
                        os.path.join(exp_path,
                                     f'resoluted_test_epoch_{epoch}_iter_{global_step}.png'),
                        normalize=True)
                netG.train()

        schedulerG.step()
        schedulerD.step()

    # Save final checkpoint
    torch.save({k: v.cpu() for k, v in netG.state_dict().items()},
               os.path.join(exp_path, 'netG_finetuned_final.pth'))
    print(f"\nFine-tuning complete.  Final checkpoint → {exp_path}/netG_finetuned_final.pth")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeepSpace foundation model fine-tuning')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--foundation_ckpt', type=str, required=True,
                        help='Path to the pretrained foundation model .pth file')

    # Dataset
    parser.add_argument('--dataset', default='deepgreen_16_256')
    parser.add_argument('--datadir', default='./data/new_region_16_256')
    parser.add_argument('--data_len', type=int, default=100,
                        help='Number of images to use from the new region (50-100 recommended)')
    parser.add_argument('--l_resolution', type=int, default=16)
    parser.add_argument('--h_resolution', type=int, default=256)

    # Training
    parser.add_argument('--exp', default='finetune_new_region')
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr_g', type=float, default=5e-5,
                        help='Generator LR (lower than foundation training)')
    parser.add_argument('--lr_d', type=float, default=2e-5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--r1_gamma', type=float, default=0.05)
    parser.add_argument('--lazy_reg', type=int, default=None)
    parser.add_argument('--rec_loss', action='store_true', default=True)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--no_lr_decay', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_ckpt_every', type=int, default=50)
    parser.add_argument('--save_content_every', type=int, default=50)

    # Model architecture (must match foundation model)
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
    parser.add_argument('--num_disc_layers', type=int, default=6)
    parser.add_argument('--no_use_fbn', action='store_true')
    parser.add_argument('--no_use_freq', action='store_true')
    parser.add_argument('--no_use_residual', action='store_true')
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--n_mlp', type=int, default=3)

    # Diffusion coefficients
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)

    args = parser.parse_args()
    finetune(args)
