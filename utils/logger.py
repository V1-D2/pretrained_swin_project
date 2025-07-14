# utils/logger.py
import os
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Логгер для отслеживания процесса обучения"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Текстовый лог
        self.log_file = os.path.join(log_dir, 'training.log')

        # Начальное сообщение
        self.log_message(f"Training started at {datetime.datetime.now()}")
        self.log_message("=" * 50)

    def log_message(self, message):
        """Записать сообщение в лог"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.datetime.now()}: {message}\n")
        print(message)

    def log_training(self, epoch, iteration, loss, psnr, ssim):
        """Логирование метрик обучения"""
        message = f"Epoch {epoch}, Iter {iteration}: Loss={loss:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}"
        self.log_message(message)

    def log_validation(self, epoch, loss, psnr, ssim):
        """Логирование метрик валидации"""
        message = f"Validation Epoch {epoch}: Loss={loss:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}"
        self.log_message(message)

    def close(self):
        """Закрыть логгер"""
        self.log_message("Training finished")
        self.log_message("=" * 50)


# utils/common.py
import os
import torch
import shutil


class AverageMeter:
    """Вычисление и хранение среднего и текущего значения"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, output_dir):
    """Сохранение checkpoint"""
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Сохраняем последний checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(state, latest_path)

    # Сохраняем лучший checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        shutil.copyfile(latest_path, best_path)

    # Сохраняем checkpoint для конкретной эпохи
    epoch = state['epoch']
    if epoch % 10 == 0:  # Каждые 10 эпох
        epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        shutil.copyfile(latest_path, epoch_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Загрузка checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    epoch = checkpoint.get('epoch', 0)
    best_psnr = checkpoint.get('best_psnr', 0)

    return epoch, best_psnr


def count_parameters(model):
    """Подсчет параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_network(model):
    """Вывод информации о сети"""
    num_params = count_parameters(model)
    print(model)
    print(f'Total number of parameters: {num_params:,}')

# utils/__init__.py
# Пустой файл для инициализации пакета