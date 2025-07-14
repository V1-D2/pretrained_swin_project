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

    # Сохраняем checkpoint каждые 10 эпох
    if state['epoch'] % 10 == 0:
        epoch_path = os.path.join(checkpoint_dir, f'epoch_{state["epoch"]}.pth')
        shutil.copyfile(latest_path, epoch_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Загрузка checkpoint"""
    checkpoint = torch.load(checkpoint_path)

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