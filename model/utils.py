# coding: utf-8
# @Author    :陈梦淇
# @time      :2022/8/21
import torch
import logging
from logging import handlers

from torch import nn
import torch.nn.functional as F


class LRScheduler:
    def __init__(
            self, optimizer, patience=10, min_lr=1e-6, factor=0.5
    ):
        """
        初始化，学习率更新策略:new_lr = old_lr * factor
        :param optimizer: 优化器
        :param patience: 容忍验证集损失不发生变化的最大轮数
        :param min_lr: 最小的学习率
        :param factor: 学习率更新因子
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0001):
        """

        :param patience:
        :param min_delta: 上一轮损失和此轮损失的阈值界限
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter = self.counter + 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # 设置日志格式

    stream_handler = logging.StreamHandler()  # 设置往屏幕上输出
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    th = logging.FileHandler(log_name)
    th.setFormatter(formatter)
    logger.addHandler(th)

    return logger


class FocalLoss(nn.Module):
    def __init__(self, weight, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduce=False)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        使用了cross_entropy
        :param inputs: (N, C)=>(样本数，类别数),without normalization
        :param targets: (N,)
        :return:
        """
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, weight, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.weight)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
