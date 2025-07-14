# data/data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import os
from typing import Dict, Tuple, List, Optional
import gc
import cv2


class TemperatureDataset(Dataset):
    """Dataset для температурных данных с MaxPool деградацией"""

    def __init__(self, npz_file: str, scale_factor: int = 4,
                 patch_height: int = 800, patch_width: int = 192,
                 max_samples: Optional[int] = None, phase: str = 'train',
                 full_size_val: bool = False):
        """
        Args:
            npz_file: путь к NPZ файлу
            scale_factor: фактор уменьшения (2 или 4)
            patch_height: высота патча для обучения
            patch_width: ширина патча для обучения
            max_samples: максимальное количество примеров
            phase: 'train' или 'val'
            full_size_val: использовать полный размер для валидации
        """
        self.npz_file = npz_file
        self.scale_factor = scale_factor
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.phase = phase
        self.full_size_val = full_size_val and phase == 'val'
        self.max_samples = max_samples

        # Загружаем данные
        print(f"Loading {npz_file}...")
        data = np.load(npz_file, allow_pickle=True)

        # Определяем ключ с данными
        if 'swaths' in data:
            self.swaths = data['swaths']
        elif 'swath_array' in data:
            self.swaths = data['swath_array']
        else:
            raise KeyError(f"Neither 'swaths' nor 'swath_array' found in {npz_file}")

        # Подготавливаем список температурных массивов
        self.temperatures = []
        self.metadata = []

        n_samples = len(self.swaths) if max_samples is None else min(len(self.swaths), max_samples)

        print(f"Preprocessing {n_samples} samples...")
        valid_samples = 0

        for i in range(len(self.swaths)):
            if valid_samples >= n_samples:
                break

            swath = self.swaths[i]
            temp = swath['temperature'].astype(np.float32)

            # Пропускаем слишком маленькие изображения
            if self.full_size_val:
                min_size = 200  # Минимальный размер для валидации
            else:
                min_size = max(patch_height, patch_width)

            if temp.shape[0] < min_size or temp.shape[1] < min_size:
                continue

            # Удаляем NaN
            mask = np.isnan(temp)
            if mask.any():
                mean_val = np.nanmean(temp)
                if np.isnan(mean_val):
                    continue
                temp[mask] = mean_val

            # Фильтрация по процентилям
            p1, p99 = np.percentile(temp, [1, 99])
            temp = np.clip(temp, p1, p99)

            # Нормализация в [0, 1]
            temp_min, temp_max = np.min(temp), np.max(temp)
            if temp_max > temp_min:
                temp_norm = (temp - temp_min) / (temp_max - temp_min)
            else:
                continue

            self.temperatures.append(temp_norm)
            self.metadata.append({
                'original_min': temp_min,
                'original_max': temp_max,
                'orbit_type': swath.get('metadata', {}).get('orbit_type', 'unknown')
            })

            valid_samples += 1

            if valid_samples % 500 == 0:
                print(f"  Processed {valid_samples}/{n_samples} samples")

        data.close()
        gc.collect()

        print(f"Loaded {len(self.temperatures)} valid samples from {npz_file}")

    def __len__(self):
        return len(self.temperatures)

    def create_lr_maxpool(self, hr: torch.Tensor) -> torch.Tensor:
        """Создание LR версии через MaxPool"""
        # Добавляем batch dimension если нужно
        if hr.dim() == 2:
            hr = hr.unsqueeze(0).unsqueeze(0)
        elif hr.dim() == 3:
            hr = hr.unsqueeze(0)

        # MaxPool для уменьшения в scale_factor раз
        if self.scale_factor == 2:
            lr = F.max_pool2d(hr, kernel_size=2, stride=2)
        elif self.scale_factor == 4:
            # Двукратное применение MaxPool2d для x4
            lr = F.max_pool2d(hr, kernel_size=2, stride=2)
            lr = F.max_pool2d(lr, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unsupported scale factor: {self.scale_factor}")

        # Убираем лишние dimensions
        return lr.squeeze(0)

    def random_crop(self, img: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Случайный кроп патча из изображения"""
        h, w = img.shape

        # Паддинг если изображение меньше нужного размера
        if h < self.patch_height or w < self.patch_width:
            pad_h = max(0, self.patch_height - h)
            pad_w = max(0, self.patch_width - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = img.shape

        # Случайные координаты
        top = np.random.randint(0, h - self.patch_height + 1)
        left = np.random.randint(0, w - self.patch_width + 1)

        return img[top:top + self.patch_height, left:left + self.patch_width], top, left

    def __getitem__(self, idx):
        # Получаем температурный массив
        temp_hr = self.temperatures[idx]
        meta = self.metadata[idx]

        if self.phase == 'train':
            # Для обучения - случайный кроп
            temp_hr_patch, _, _ = self.random_crop(temp_hr)

            # Убеждаемся, что размеры кратны scale_factor
            h, w = temp_hr_patch.shape
            h = h - h % self.scale_factor
            w = w - w % self.scale_factor
            temp_hr_patch = temp_hr_patch[:h, :w]

            # Конвертируем в тензор
            hr_tensor = torch.from_numpy(temp_hr_patch).unsqueeze(0).float()

            # Создаем LR через MaxPool
            lr_tensor = self.create_lr_maxpool(hr_tensor)

        else:  # validation
            if self.full_size_val:
                # Используем полное изображение для валидации
                h, w = temp_hr.shape
                # Обрезаем до кратности scale_factor
                h = h - h % self.scale_factor
                w = w - w % self.scale_factor
                temp_hr_patch = temp_hr[:h, :w]
            else:
                # Центральный кроп для валидации
                h, w = temp_hr.shape
                if h > self.patch_height and w > self.patch_width:
                    top = (h - self.patch_height) // 2
                    left = (w - self.patch_width) // 2
                    temp_hr_patch = temp_hr[top:top + self.patch_height,
                                    left:left + self.patch_width]
                else:
                    temp_hr_patch = temp_hr
                    h, w = temp_hr_patch.shape
                    h = h - h % self.scale_factor
                    w = w - w % self.scale_factor
                    temp_hr_patch = temp_hr_patch[:h, :w]

            # Конвертируем в тензор
            hr_tensor = torch.from_numpy(temp_hr_patch).unsqueeze(0).float()

            # Создаем LR через MaxPool
            lr_tensor = self.create_lr_maxpool(hr_tensor)

        return {
            'lq': lr_tensor,  # low quality (low resolution)
            'gt': hr_tensor,  # ground truth (high resolution)
            'lq_path': f'{self.npz_file}_{idx}',
            'gt_path': f'{self.npz_file}_{idx}',
            'metadata': meta
        }


def create_train_val_dataloaders(train_files: List[str], val_file: str,
                                 batch_size: int = 4, scale_factor: int = 4,
                                 patch_height: int = 800, patch_width: int = 192,
                                 val_samples: int = 10,
                                 train_samples_per_file: int = 3000) -> Tuple[DataLoader, DataLoader]:
    """Создание train и validation датлоадеров"""

    # Training datasets - загружаем последовательно для экономии памяти
    train_datasets = []

    for train_file in train_files:
        if os.path.exists(train_file):
            dataset = TemperatureDataset(
                train_file,
                scale_factor=scale_factor,
                patch_height=patch_height,
                patch_width=patch_width,
                max_samples=train_samples_per_file,
                phase='train'
            )
            train_datasets.append(dataset)
            print(f"Added {len(dataset)} samples from {train_file}")

    # Объединяем все training datasets
    train_dataset = ConcatDataset(train_datasets)

    # Training dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    # Validation dataset - полноразмерные изображения
    val_dataset = TemperatureDataset(
        val_file,
        scale_factor=scale_factor,
        patch_height=patch_height,
        patch_width=patch_width,
        max_samples=val_samples,
        phase='val',
        full_size_val=True  # Используем полный размер для валидации
    )

    # Validation dataloader - batch_size=1 для полноразмерных изображений
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"\nCreated dataloaders:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} full-size samples")

    return train_loader, val_loader


if __name__ == "__main__":
    # Тестирование
    import matplotlib.pyplot as plt

    # Создаем тестовый dataset
    dataset = TemperatureDataset(
        "test_data.npz",
        scale_factor=4,
        patch_height=800,
        patch_width=192,
        max_samples=5,
        phase='train'
    )

    # Получаем один пример
    sample = dataset[0]
    lr = sample['lq'].numpy()[0]
    hr = sample['gt'].numpy()[0]

    print(f"LR shape: {lr.shape}")
    print(f"HR shape: {hr.shape}")
    print(f"LR range: [{lr.min():.3f}, {lr.max():.3f}]")
    print(f"HR range: [{hr.min():.3f}, {hr.max():.3f}]")

    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(lr, cmap='viridis', aspect='auto')
    ax1.set_title(f'LR ({lr.shape[0]}x{lr.shape[1]})')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(hr, cmap='viridis', aspect='auto')
    ax2.set_title(f'HR ({hr.shape[0]}x{hr.shape[1]})')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig('test_maxpool_degradation.png')
    print("Saved visualization to test_maxpool_degradation.png")