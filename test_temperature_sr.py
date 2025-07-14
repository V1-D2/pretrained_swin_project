#!/usr/bin/env python3
"""
Test script for Temperature Super-Resolution Model
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from models.network_swinir import SwinIR
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim
import torch.nn.functional as F


def load_model(checkpoint_path, device, scale_factor=4):
    """Load trained model from checkpoint"""

    # Create model
    model = SwinIR(
        upscale=scale_factor,
        in_chans=1,
        img_size=(800, 192),  # Should match training
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='1conv'
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def test_single_image(model, lr_image, device, window_size=8):
    """Test model on single image"""

    # Prepare input
    if isinstance(lr_image, np.ndarray):
        lr_tensor = torch.from_numpy(lr_image).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        lr_tensor = lr_image.to(device)

    # Pad to multiple of window_size
    _, _, h, w = lr_tensor.shape
    h_pad = (window_size - h % window_size) % window_size
    w_pad = (window_size - w % window_size) % window_size
    lr_padded = F.pad(lr_tensor, (0, w_pad, 0, h_pad), mode='reflect')

    # Forward pass
    with torch.no_grad():
        sr_padded = model(lr_padded)

    # Remove padding
    sr = sr_padded[:, :, :h * model.upscale, :w * model.upscale]

    return sr.squeeze().cpu().numpy()


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device, args.scale_factor)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    print(f"Loading test data from {args.test_npz}")
    data = np.load(args.test_npz, allow_pickle=True)

    if 'swaths' in data:
        swaths = data['swaths']
    elif 'swath_array' in data:
        swaths = data['swath_array']
    else:
        raise KeyError("Cannot find temperature data in NPZ file")

    # Test on samples
    num_samples = min(args.num_samples, len(swaths))
    print(f"Testing on {num_samples} samples")

    all_psnr = []
    all_ssim = []

    for i in tqdm(range(num_samples), desc="Testing"):
        swath = swaths[i]
        temp = swath['temperature'].astype(np.float32)

        # Preprocess
        # Remove NaN
        mask = np.isnan(temp)
        if mask.any():
            mean_val = np.nanmean(temp)
            temp[mask] = mean_val

        # Clip percentiles
        p1, p99 = np.percentile(temp, [1, 99])
        temp = np.clip(temp, p1, p99)

        # Normalize
        temp_min, temp_max = np.min(temp), np.max(temp)
        if temp_max > temp_min:
            temp_norm = (temp - temp_min) / (temp_max - temp_min)
        else:
            continue

        # Make sure size is divisible by scale_factor
        h, w = temp_norm.shape
        h = h - h % args.scale_factor
        w = w - w % args.scale_factor
        hr = temp_norm[:h, :w]

        # Create LR using MaxPool
        hr_tensor = torch.from_numpy(hr).float().unsqueeze(0).unsqueeze(0)
        if args.scale_factor == 2:
            lr_tensor = F.max_pool2d(hr_tensor, kernel_size=2, stride=2)
        elif args.scale_factor == 4:
            lr_tensor = F.max_pool2d(hr_tensor, kernel_size=2, stride=2)
            lr_tensor = F.max_pool2d(lr_tensor, kernel_size=2, stride=2)
        lr = lr_tensor.squeeze().numpy()

        # Test model
        sr = test_single_image(model, lr, device)

        # Calculate metrics
        hr_uint8 = (hr * 255).clip(0, 255).astype(np.uint8)
        sr_uint8 = (sr * 255).clip(0, 255).astype(np.uint8)

        psnr = calculate_psnr(sr_uint8, hr_uint8, crop_border=0)
        ssim = calculate_ssim(sr_uint8, hr_uint8, crop_border=0)

        all_psnr.append(psnr)
        all_ssim.append(ssim)

        # Save sample images
        if i < args.save_samples:
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # LR upsampled
            lr_up = cv2.resize(lr, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_NEAREST)

            im1 = axes[0].imshow(lr_up, cmap='viridis', aspect='auto')
            axes[0].set_title(f'LR Upsampled ({lr.shape[0]}×{lr.shape[1]})')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046)

            im2 = axes[1].imshow(sr, cmap='viridis', aspect='auto')
            axes[1].set_title(f'SR Output ({sr.shape[0]}×{sr.shape[1]})')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046)

            im3 = axes[2].imshow(hr, cmap='viridis', aspect='auto')
            axes[2].set_title(f'HR Ground Truth ({hr.shape[0]}×{hr.shape[1]})')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], fraction=0.046)

            plt.suptitle(f'Sample {i + 1} - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'sample_{i + 1}.png'), dpi=150)
            plt.close()

            # Save numpy arrays
            np.savez(
                os.path.join(args.output_dir, f'result_{i + 1}.npz'),
                lr=lr,
                sr=sr,
                hr=hr,
                psnr=psnr,
                ssim=ssim,
                temp_min=temp_min,
                temp_max=temp_max
            )

    # Print results
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("=" * 50)

    # Save results
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test on: {args.test_npz}\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Scale factor: {args.scale_factor}\n\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n\n")
        f.write("Individual results:\n")
        for i, (psnr, ssim) in enumerate(zip(all_psnr, all_ssim)):
            f.write(f"Sample {i + 1}: PSNR={psnr:.2f}, SSIM={ssim:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Temperature SR Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_npz', type=str, required=True,
                        help='Path to test NPZ file')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Output directory for results')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='Super-resolution scale factor')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to test')
    parser.add_argument('--save_samples', type=int, default=5,
                        help='Number of sample visualizations to save')

    args = parser.parse_args()
    main(args)