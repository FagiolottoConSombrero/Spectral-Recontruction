from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import math, random
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from signal_dataloader import SignalFlourFolderDataset


class AverageMeter(object):
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


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = error.reshape(-1).mean()
        return mrae


class Loss_MRAE_custom(nn.Module):
    def __init__(self):
        super(Loss_MRAE_custom, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        mask = label == 0
        if mask.any():
            label_wo_zero = label.clone()
            label_wo_zero[mask] = 1e-8
        else:
            label_wo_zero = label
        error = torch.abs(outputs - label) / label_wo_zero
        mrae = torch.mean(error)
        return mrae


class Loss_RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        outputs = outputs.float()
        label   = label.float()
        error = outputs - label
        mse = torch.mean(error ** 2)
        rmse = torch.sqrt(mse)
        return rmse


class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close


def make_loaders(root, sensor_root, rgb, ir, patch_mean, batch_size=8, num_workers=4, val_ratio=0.2, pin_memory=True):
    """
    Se esistono /train e /val sotto root li usa; altrimenti fa split random.
    """
    root = Path(root)
    full = SignalFlourFolderDataset(root=root,
                              spectral_sens_csv=sensor_root,
                              rgb=rgb, ir=ir,
                              hsi_channels_first=False,  # True se i tuoi HSI sono (L,H,W)
                              illuminant_mode="planck",  # alogena
                              illuminant_T=2856.0,
                              patch_mean=patch_mean)
    n = len(full)
    n_val = int(math.floor(n * val_ratio))
    n_train = n - n_val
    train_set, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    pw = num_workers > 0

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=pw
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=pw
    )
    return train_loader, val_loader


def spectral_angle_loss(yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # SAM in radianti (per logging lo convertiamo in gradi)
    num = (yhat * y).sum(dim=1)
    den = yhat.norm(dim=1) * y.norm(dim=1) + eps
    cos = (num / den).clamp(-1 + 1e-6, 1 - 1e-6)
    ang = torch.acos(cos)
    return ang.mean()


def spectral_smoothness(y: torch.Tensor) -> torch.Tensor:
    # y: (B,L) oppure (L,) -> penalizza variazioni lungo λ
    if y.dim() == 1:
        y = y.unsqueeze(0)
    diff = y[:, 1:] - y[:, :-1]
    return (diff**2).mean()