# coding: utf-8
# @Author    :陈梦淇
# @time      :2023/12/13
import os
import sys
import time
import datetime

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

current_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
sys.path.append(parent_directory)
from Ensemble_model import EnsembleModel
from utils import LRScheduler, EarlyStopping, get_logger, FocalLoss, WeightedFocalLoss
from train_utils import train_for_single_epoch, validation_for_single_epoch, test_for_single_epoch
from train_utils import train_for_multi_epoch, validation_for_multi_epoch, test_for_multi_epoch

seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

DATE_TIME = datetime.datetime.now().strftime("%m-%d")


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round(elapsed))

    return str(datetime.timedelta(seconds=elapsed_rounded))


class EMTrainer(nn.Module):
    def __init__(self, input_size,
                 learning_rate,
                 optimizer,
                 epoch_nums,
                 use_crf=False,
                 early_stop=True, device=0, description="", loss_fn="CrossEntropyLoss",
                 binding_mode="sum", downstream_mode="Linear", model_name="BiLSTM",
                 weight=1.0, is_save=False, alpha=0.25, gamma=2):
        super(EMTrainer, self).__init__()
        self.input_size = input_size
        self.epoch_nums = epoch_nums
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.early_stop = early_stop
        self.device = device
        self.description = description
        self.use_crf = use_crf
        self.is_save = is_save
        self.model_name = model_name
        self.downstream_mode = downstream_mode
        self.gamma = gamma

        # 根据日期创建目录用于保存日志文件和csv结果文件
        if not os.path.exists(f"{DATE_TIME}"):
            os.mkdir("{}".format(DATE_TIME))

        self.csv_result = pd.DataFrame(
            columns=["name", "epoch", "precision", "recall", "accuracy", "F1", "auc", "auprc",
                     "mcc", "ppv", "npv", "tpr", "tnr"])

        # cuda
        self.device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        # initialize model
        # self.model = EnsembleModel(input_size=input_size, output_size=1, binding_mode=binding_mode,
        #                            residual_mode=residual_mode)
        # self.model = BERTFineTuningWithBio(58)
        self.model = EnsembleModelMultiFea2(bio_size=58, device=self.device)
        # self.model = EnsembleModel2(input_size=58, device=self.device, downstream_mode=downstream_mode)
        # self.model = EnsembleModelMultiFea2(bio_size=58, device=self.device)
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=100,
            last_epoch=-1,
        )

        # 损失函数
        self.loss_fn = None
        if loss_fn == "CrossEntropyLoss":
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0]).to(self.device))  # 处理标签不平衡
        elif loss_fn == "FocalLoss":
            self.loss_fn = FocalLoss(weight=torch.tensor(weight).to(self.device), alpha=1,
                                     gamma=2).to(self.device)
        elif loss_fn == "BCEWithLogitsLoss":
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0).to(self.device), reduce=False)
        elif loss_fn == "WeightedFocalLoss":
            self.loss_fn = WeightedFocalLoss(weight=torch.tensor(weight).to(self.device),
                                             alpha=alpha, gamma=gamma).to(self.device)

        self.log_name = f"{DATE_TIME}/EnsembleModelMultiFea2_{DATE_TIME}_{self.model_name}_gamma={gamma}_{description}.log"
        self.logger = get_logger(self.log_name)
        self.logger.info(
            f"EnsembleModelMultiFea2，损失函数{loss_fn}，gamma:{self.gamma}_model_{self.model_name}")

        self.model = self.model.to(self.device)
        self.early_stop = True
        self.early_stopping = EarlyStopping()

    def train_and_validation_part(self, train_dataloader, validation_dataloader, test_dataloader):
        train_acc, train_loss, best_val_loss = 0, 0, 100
        best_auprc = 0
        t0 = time.time()
        self.logger.info(f"==================description:{self.description}====================")
        self.logger.info("========learning_rate:{:}=========".format(self.learning_rate))
        for epoch in range(self.epoch_nums):
            torch.cuda.empty_cache()
            lr = self.scheduler.get_last_lr()[0]
            self.logger.info("")
            self.logger.info('======== Epoch {:} / {:}, lr:{:} ========'.format(epoch + 1, self.epoch_nums, lr))
            self.logger.info('Training...')

            train_loss = train_for_multi_epoch(self.model, train_dataloader, self.optimizer, self.device, self.logger,
                                               self.use_crf, self.loss_fn)
            val_loss = validation_for_multi_epoch(self.model, validation_dataloader, self.device, self.logger, epoch,
                                                  self.use_crf, self.loss_fn, self.csv_result)

            # 如果模型在验证集上表现效果不错，则测试其在盲测集上的性能
            auprc_epoch = test_for_multi_epoch(self.model, test_dataloader, self.device, self.logger, self.use_crf,
                                               self.csv_result, epoch)
            self.save_model(epoch)

            # if auprc_epoch > best_auprc and self.is_save:
            #     best_auprc = auprc_epoch
            #     self.save_model(epoch)
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #
            #     # self.save_model(epoch)
            self.scheduler.step()
            if self.early_stop:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    break

        self.logger.info("Training complete!")
        self.logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - t0)))

        self.logger.info("==========train loss: {}==========".format(train_loss))
        self.logger.info("==========validation loss: {}==========".format(best_val_loss))

        # self.save_model(self.epoch_nums)
        self.csv_result.to_csv(f"{self.log_name}.csv")
        return best_auprc

    def save_model(self, epoch):
        if not os.path.exists("{}_model".format(DATE_TIME)):
            os.mkdir("{}_model".format(DATE_TIME))
        torch.save(self.model.state_dict(),
                   "{}_model/EnsembleModelMultiFea2_{}_{}_{}_{}_{}.pth".format(DATE_TIME, DATE_TIME,
                                                                               self.downstream_mode,
                                                                               self.optimizer_type,
                                                                               epoch,
                                                                               self.gamma,
                                                                               self.description))
