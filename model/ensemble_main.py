# coding: utf-8
# @Author    :陈梦淇
# @time      :2023/12/13
import os
import sys
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
import optuna

from Ensemble_train import EMTrainer

current_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
sys.path.append(parent_directory)
from data_utils import collate_fn_for_bio, PPISDataForBio
from data_utils import collate_fn_for_bertFreeze, PPISDataForBERTFreeze
from data_utils import collate_fn_for_BERTFreezeBio, PPISDataForBERTFreezeBio

seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
date_time = datetime.datetime.now().strftime("%m-%d")


def load_data(group, file_name, file_format, id_file_name, params_dict):
    ppis_data = params_dict["dataloader_fn"](data_group=group, file_name=file_name, file_format=file_format,
                                             id_file_name=id_file_name, all_fea=False, fea_num=params_dict["bio_num"])
    ppis_dataloader = DataLoader(ppis_data, batch_size=params_dict["batch_size"], shuffle=False, pin_memory=True,
                                 num_workers=3, collate_fn=params_dict["collate_fn"])
    return ppis_dataloader


def get_data(batch_size, bio_num, dataloader_fn, collate_fn):
    data_load_param = {
        "batch_size": batch_size,
        "dataloader_fn": dataloader_fn,
        "collate_fn": collate_fn,
        "bio_num": bio_num
    }

    train_dataloader = load_data(group="DEPHI", file_name="DEPHI_train_th_aug",
                                 id_file_name="DEPHI_train_th_aug_id",
                                 file_format="pkl", params_dict=data_load_param)
    # train_dataloader = load_data(group="DEPHI", file_name="DEPHI_validation_th",
    #                              id_file_name="DEPHI_validation_th_id",
    #                              file_format="pkl", params_dict=data_load_param)

    validation_dataloader = load_data(group="DEPHI", file_name="DEPHI_validation_th",
                                      id_file_name="DEPHI_validation_th_id",
                                      file_format="pkl", params_dict=data_load_param)

    # test_72_dataloader = load_data(group="EDLMPPI", file_name="seq_72", id_file_name="seq_72_id",
    #                                file_format="csv", params_dict=data_load_param)

    # test_164_dataloader = load_data(group="DSet164", file_name="DSet164", id_file_name="DSet164_id",
    #                                 file_format="pkl", params_dict=data_load_param)
    test_186_dataloader = load_data(group="DSet186", file_name="DSet186", id_file_name="DSet186_id",
                                    file_format="pkl", params_dict=data_load_param)
    # test_355_dataloader = load_data(group="DSet355", file_name="DSet355", id_file_name="DSet_355_id",
    #                                 file_format="pkl", params_dict=data_load_param)
    # test_448_dataloader = load_data(group="DSet448", file_name="DSet448", id_file_name="DSet448_id",
    #                                 file_format="pkl", params_dict=data_load_param)
    # test_60_dataloader = load_data(group="ours", file_name="Test_60_predict", id_file_name="Test_60_predict_id",
    #                                file_format="csv", params_dict=data_load_param)
    # test_DEPHI_dataloader = load_data(group="DEPHI", file_name="DEPHI_test_110_truncated", id_file_name="DEPHI_test_id",
    #                                   file_format="pkl", params_dict=data_load_param)

    test_dataloader_dict = {
        "Dset_186": test_186_dataloader,
        # "Dset_164": test_164_dataloader,
        # "blindTest": test_DEPHI_dataloader,
        # "DSet448": test_448_dataloader
    }

    dataset_loader_dict = {
        "train_dataloader": train_dataloader,
        "validation_dataloader": validation_dataloader,
        "test_dataloader": test_dataloader_dict
    }
    return dataset_loader_dict


def get_data_test(batch_size, dataloader_fn, collate_fn):
    train_dataloader = load_data(group="EDLMPPI", file_name="seq_72_fea", batch_size=batch_size,
                                 dataloader_fn=dataloader_fn, collate_fn=collate_fn)
    validation_dataloader = load_data(group="EDLMPPI", file_name="seq_72_fea", batch_size=batch_size,
                                      dataloader_fn=dataloader_fn, collate_fn=collate_fn)
    test_dataloader = load_data(group="EDLMPPI", file_name="seq_72_fea", batch_size=batch_size,
                                dataloader_fn=dataloader_fn, collate_fn=collate_fn)
    test_dataloader_dict = {
        "test_dataloader": test_dataloader
    }

    dataset_loader_dict = {
        "train_dataloader": train_dataloader,
        "validation_dataloader": validation_dataloader,
        "test_dataloader": test_dataloader_dict
    }
    return dataset_loader_dict


def blg_main(para_args):
    """
    训练BERT_LSTM_GRAPH Model
    :param para_args:
    :return:
    """
    dataset_loader = get_data(para_args.batch_size, -1, PPISDataForBERTFreezeBio, collate_fn_for_BERTFreezeBio)
    gamma = [1, 2, 3, 4, 5]
    for g in gamma:
        em_trainer = EMTrainer(
            input_size=para_args.features_num,
            learning_rate=para_args.learning_rate,
            optimizer=para_args.optimizer,
            epoch_nums=para_args.epoch,
            description=para_args.description,
            loss_fn="BCEWithLogitsLoss",
            binding_mode="sum",
            model_name="Transformer",
            weight=1,
            is_save=True,
            downstream_mode="Capsule",
            gamma=g
        )
        auprc = em_trainer.train_and_validation_part(train_dataloader=dataset_loader["train_dataloader"],
                                                     validation_dataloader=dataset_loader["validation_dataloader"],
                                                     test_dataloader=dataset_loader["test_dataloader"])


    # for w in weight:
    #     em_trainer = EMTrainer(
    #         input_size=para_args.features_num,
    #         learning_rate=para_args.learning_rate,
    #         optimizer=para_args.optimizer,
    #         epoch_nums=para_args.epoch,
    #         description=para_args.description,
    #         loss_fn="WeightedFocalLoss",
    #         binding_mode="sum",
    #         weight=w,
    #         is_save=False,
    #         downstream_mode="Capsule"
    #     )
    #     auprc = em_trainer.train_and_validation_part(train_dataloader=dataset_loader["train_dataloader"],
    #                                                  validation_dataloader=dataset_loader["validation_dataloader"],
    #                                                  test_dataloader=dataset_loader["test_dataloader"])
    #
    # return auprc


def objective(trial):
    w = trial.suggest_float('w', 1.0, 2.0)
    em_trainer = EMTrainer(
        input_size=args.features_num,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        epoch_nums=args.epoch,
        description=args.description,
        loss_fn="FocalLoss",
        binding_mode="sum",
        model_name="Transformer",
        weight=w,
        is_save=False
    )
    auprc = em_trainer.train_and_validation_part(train_dataloader=dataset_loader["train_dataloader"],
                                                 validation_dataloader=dataset_loader["validation_dataloader"],
                                                 test_dataloader=dataset_loader["test_dataloader"])
    return auprc


if __name__ == "__main__":
    print("运行日期为:{}".format(date_time))
    parser = argparse.ArgumentParser(description="显示程序搜索的超参数")
    parser.add_argument("--optimizer", type=str, help="优化器类型", default="AdamW")
    parser.add_argument("--learning_rate", type=float, help="学习率", default=2e-4)
    parser.add_argument("--epoch", type=int, help="训练轮数", default=200)
    parser.add_argument("--features_num", type=int, help="使用特征的数量", default=58)
    parser.add_argument("--device", type=int, help="gpu序号", default=0)
    parser.add_argument("--description", type=str, help="说明", default="")
    args = parser.parse_args()
    print(f"description: {args.description}")
    args.batch_size = 6
    # dataset_loader = get_data(args.batch_size, 64, PPISDataForBio, collate_fn_for_bio)
    # study = optuna.create_study(study_name='Transformer_weight', direction='maximize', sampler=optuna.samplers.TPESampler())
    # study.optimize(objective, n_trials=50)
    # print("Best trial:")
    # trial = study.best_trial
    # print("  Value: ", trial.value)
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("{}:{}".format(key, value))
    blg_main(args)
