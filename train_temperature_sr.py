#!/usr/bin/env python3
"""
Temperature Super-Resolution Training Script
Trains SwinIR model for single-channel temperature data enhancement
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import time
from datetime import datetime
import json
from tqdm import tqdm
import cv2
import gc
from collections import OrderedDict

# Import from project
from models.network_swinir import SwinIR
from data.data_loader import create_train_val_dataloaders
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim
from utils.logger import Logger
from utils.common import AverageMeter, save_checkpoint, load_checkpoint


def adapt_pretrained_weights(pretrained_path, model, device):
    """
    Адаптация весов 3-канальной модели для 1-канальной
    """
    print("Loading and adapting pretrained weights...")

    # Загружаем веса
    pretrained = torch.load(pretrained_path, map_location=device)

    # Извлекаем state_dict
    if isinstance(pretrained, dict):
        if 'params' in pretrained:
            pretrained_dict = pretrained['params']
        elif 'state_dict' in pretrained:
            pretrained_dict = pretrained['state_dict']
        else:
            pretrained_dict = pretrained
    else:
        pretrained_dict = pretrained

    # Получаем текущий state_dict модели
    model_dict = model.state_dict()

    # Адаптируем веса первого слоя (conv_first)
    if 'conv_first.weight' in pretrained_dict:
        conv_first_weight = pretrained_dict['conv_first.weight']
        # Усредняем веса по каналам: (out_channels, 3, H, W) -> (out_channels, 1, H, W)
        conv_first_weight_adapted = conv_first_weight.mean(dim=1, keepdim=True)
        pretrained_dict['conv_first.weight'] = conv_first_weight_adapted
        print(f"Adapted conv_first: {conv_first_weight.shape} -> {conv_first_weight_adapted.shape}")

    # Адаптируем веса последнего слоя (conv_last)
    if 'conv_last.weight' in pretrained_dict:
        conv_last_weight = pretrained_dict['conv_last.weight']
        # Берем только первый выходной канал: (3, in_channels, H, W) -> (1, in_channels, H, W)
        conv_last_weight_adapted = conv_last_weight[0:1, :, :, :]
        pretrained_dict['conv_last.weight'] = conv_last_weight_adapted

        # Адаптируем bias
        if 'conv_last.bias' in pretrained_dict:
            conv_last_bias = pretrained_dict['conv_last.bias']
            conv_last_bias_adapted = conv_last_bias[0:1]
            pretrained_dict['conv_last.bias'] = conv_last_bias_adapted

        print(f"Adapted conv_last: {conv_last_weight.shape} -> {conv_last_weight_adapted.shape}")

    # Фильтруем только те веса, которые подходят по размеру
    adapted_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                adapted_dict[k] = v
            else:
                print(f"Skipping {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")

    # Обновляем веса модели
    model_dict.update(adapted_dict)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(adapted_dict)}/{len(model_dict)} layers from pretrained model")
    return model


def percentile_clip(tensor, lower=1, upper=99):
    """Обрезка значений по процентилям для устранения выбросов"""
    lower_val = torch.quantile(tensor, lower / 100.0)
    upper_val = torch.quantile(tensor, upper / 100.0)
    return torch.clamp(tensor, lower_val, upper_val)


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, logger, device, writer):
    """Обучение одной эпохи"""
    model.train()

    losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()

    # Прогресс бар
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for i, batch in enumerate(pbar):
        # Загружаем данные
        lq = batch['lq'].to(device)  # Low quality (LR)
        gt = batch['gt'].to(device)  # Ground truth (HR)

        # Фильтрация выбросов
        lq = percentile_clip(lq, 1, 99)
        gt = percentile_clip(gt, 1, 99)

        # Forward pass
        sr = model(lq)
        loss = criterion(sr, gt)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Вычисляем метрики
        with torch.no_grad():
            # Конвертируем в numpy для метрик
            sr_np = sr.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()

            psnr_val = 0
            ssim_val = 0

            # Вычисляем PSNR и SSIM для каждого изображения в батче
            for j in range(sr_np.shape[0]):
                sr_img = (sr_np[j, 0] * 255).clip(0, 255).astype(np.uint8)
                gt_img = (gt_np[j, 0] * 255).clip(0, 255).astype(np.uint8)

                psnr_val += calculate_psnr(sr_img, gt_img, crop_border=0)
                ssim_val += calculate_ssim(sr_img, gt_img, crop_border=0)

            psnr_val /= sr_np.shape[0]
            ssim_val /= sr_np.shape[0]

        # Обновляем метрики
        losses.update(loss.item(), lq.size(0))
        psnrs.update(psnr_val, lq.size(0))
        ssims.update(ssim_val, lq.size(0))

        # Обновляем прогресс бар
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'PSNR': f'{psnrs.avg:.2f}',
            'SSIM': f'{ssims.avg:.4f}'
        })

        # Логирование в TensorBoard каждые 50 итераций
        global_step = epoch * len(train_loader) + i
        if i % 50 == 0:
            writer.add_scalar('Train/Loss', losses.val, global_step)
            writer.add_scalar('Train/PSNR', psnrs.val, global_step)
            writer.add_scalar('Train/SSIM', ssims.val, global_step)

        # Очистка памяти
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    return losses.avg, psnrs.avg, ssims.avg


def validate(model, val_loader, criterion, epoch, logger, device, writer, save_dir=None):
    """Валидация модели на полноразмерных изображениях"""
    model.eval()

    losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()

    # Создаем директорию для сохранения примеров
    if save_dir:
        sample_dir = os.path.join(save_dir, 'samples', f'epoch_{epoch}')
        os.makedirs(sample_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation {epoch}')

        for i, batch in enumerate(pbar):
            lq = batch['lq'].to(device)
            gt = batch['gt'].to(device)

            # Обработка полноразмерного изображения
            # Паддинг для кратности window_size
            _, _, h, w = lq.shape
            window_size = 8
            scale = 4

            # Паддинг LR
            h_pad = (window_size - h % window_size) % window_size
            w_pad = (window_size - w % window_size) % window_size
            lq_padded = torch.nn.functional.pad(lq, (0, w_pad, 0, h_pad), mode='reflect')

            # Forward pass
            sr = model(lq_padded)

            # Убираем паддинг из SR
            sr = sr[:, :, :h * scale, :w * scale]

            # Loss
            loss = criterion(sr, gt)

            # Метрики
            sr_np = sr.cpu().numpy()[0, 0]
            gt_np = gt.cpu().numpy()[0, 0]
            lq_np = lq.cpu().numpy()[0, 0]

            sr_img = (sr_np * 255).clip(0, 255).astype(np.uint8)
            gt_img = (gt_np * 255).clip(0, 255).astype(np.uint8)

            psnr_val = calculate_psnr(sr_img, gt_img, crop_border=0)
            ssim_val = calculate_ssim(sr_img, gt_img, crop_border=0)

            losses.update(loss.item(), 1)
            psnrs.update(psnr_val, 1)
            ssims.update(ssim_val, 1)

            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'PSNR': f'{psnrs.avg:.2f}',
                'SSIM': f'{ssims.avg:.4f}'
            })

            # Сохраняем первые 3 примера
            if save_dir and i < 3:
                # Апсэмплинг LR для визуализации
                lq_up = cv2.resize(lq_np, (sr_np.shape[1], sr_np.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

                # Нормализуем для визуализации
                def normalize_for_vis(img):
                    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    return (img_norm * 255).astype(np.uint8)

                lq_vis = normalize_for_vis(lq_up)
                sr_vis = normalize_for_vis(sr_np)
                gt_vis = normalize_for_vis(gt_np)

                # Создаем композитное изображение
                composite = np.hstack([lq_vis, sr_vis, gt_vis])

                # Добавляем подписи
                h_comp, w_comp = composite.shape
                composite_labeled = np.ones((h_comp + 50, w_comp), dtype=np.uint8) * 255
                composite_labeled[:h_comp, :] = composite

                # Сохраняем
                cv2.imwrite(os.path.join(sample_dir, f'sample_{i + 1}_composite.png'),
                            composite_labeled)

                # Сохраняем отдельные изображения
                cv2.imwrite(os.path.join(sample_dir, f'sample_{i + 1}_lr.png'), lq_vis)
                cv2.imwrite(os.path.join(sample_dir, f'sample_{i + 1}_sr.png'), sr_vis)
                cv2.imwrite(os.path.join(sample_dir, f'sample_{i + 1}_hr.png'), gt_vis)

    # Логирование в TensorBoard
    writer.add_scalar('Val/Loss', losses.avg, epoch)
    writer.add_scalar('Val/PSNR', psnrs.avg, epoch)
    writer.add_scalar('Val/SSIM', ssims.avg, epoch)

    logger.log_validation(epoch, losses.avg, psnrs.avg, ssims.avg)

    # Очистка памяти
    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg, psnrs.avg, ssims.avg


def main(args):
    """Основная функция обучения"""
    # Создаем директории
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)

    # Инициализируем логгер и TensorBoard
    logger = Logger(os.path.join(args.output_dir, 'logs'))
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))

    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Создаем датасеты
    train_files = [
        os.path.join(args.data_dir, f'temperature_data_{i}.npz')
        for i in range(1, 4)  # Используем файлы 1-3 для обучения
    ]
    val_file = os.path.join(args.data_dir, 'temperature_data_4.npz')  # Файл 4 для валидации

    print("Creating data loaders...")
    train_loader, val_loader = create_train_val_dataloaders(
        train_files,
        val_file,
        batch_size=args.batch_size,
        scale_factor=args.scale_factor,
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        val_samples=10,  # 10 полноразмерных изображений для валидации
        train_samples_per_file=4000  # Ограничиваем для экономии памяти
    )

    # Создаем модель
    print("Creating model...")
    model = SwinIR(
        upscale=args.scale_factor,
        in_chans=1,  # Одноканальный ввод
        img_size=(args.patch_height, args.patch_width),
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='1conv'
    ).to(device)

    # Загружаем и адаптируем предобученную модель
    if args.pretrained:
        model = adapt_pretrained_weights(args.pretrained, model, device)

    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function и optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-7
    )

    # Загружаем checkpoint если есть
    start_epoch = 0
    best_psnr = 0

    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'latest.pth')
        if os.path.exists(checkpoint_path):
            start_epoch, best_psnr = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
            print(f"Resumed from epoch {start_epoch} with best PSNR {best_psnr:.2f}")

    # Сохраняем конфигурацию
    config = vars(args)
    config['model_params'] = total_params
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Основной цикл обучения
    print("\nStarting training...")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'=' * 50}")

        # Обучение
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch + 1, logger, device, writer
        )

        # Валидация
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, criterion, epoch + 1, logger, device, writer, args.output_dir
        )

        # Обновляем learning rate
        scheduler.step()

        # Сохраняем checkpoint
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        # Сохраняем каждые 10 эпох и лучшую модель
        if (epoch + 1) % 10 == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'args': args
            }, is_best, args.output_dir)

        # Выводим итоги эпохи
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        print(f"  Best PSNR: {best_psnr:.2f}")

        # Очистка памяти после каждой эпохи
        torch.cuda.empty_cache()
        gc.collect()

    print("\nTraining completed!")
    print(f"Best validation PSNR: {best_psnr:.2f}")

    writer.close()
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temperature SR Training')

    # Paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory with NPZ files')
    parser.add_argument('--output_dir', type=str, default='./experiments/temperature_sr',
                        help='Path to save outputs')
    parser.add_argument('--pretrained', type=str,
                        default='./pretrained/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth',
                        help='Path to pretrained model')

    # Model parameters
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='Super-resolution scale factor')
    parser.add_argument('--patch_height', type=int, default=800,
                        help='Training patch height')
    parser.add_argument('--patch_width', type=int, default=192,
                        help='Training patch width')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')

    # Distributed training
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')

    args = parser.parse_args()

    # Setup distributed training if needed
    if args.world_size > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    main(args)