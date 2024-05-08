# coding: utf-8
# @Author    :陈梦淇
# @time      :2023/11/10
import os
import sys
import time
import datetime

import torch
import numpy as np

current_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_directory)
from metrics import metrics_classification
from crf_utils import viterbi_decode


def get_label_from_mask(labels, attention_masks):
    """

    :param labels:
    :param attention_masks:
    :return: tensor(N,)
    """
    y_true = []
    for seq_num in range(len(labels)):
        seq_len = (attention_masks[seq_num] == 1).sum()
        label = labels[seq_num][0:seq_len]
        y_true.append(label)
    y_true = torch.cat(y_true, 0).view(-1, )
    return y_true


def get_label_list_from_mask(labels, attention_masks, source):
    """
    不拼接
    :param labels:
    :param attention_masks:
    :param source: predict or real
    :return:
    """
    y_original, length_list = [], []
    labels = labels.detach().cpu().numpy().tolist()

    if source == "real":
        for seq_num in range(len(labels)):
            seq_len = (attention_masks[seq_num] == 1).sum()
            label = labels[seq_num][0:seq_len]
            y_original.append(label)
        return y_original
    elif source == "predict":
        for seq_num in range(attention_masks.shape[0]):
            seq_len = (attention_masks[seq_num] == 1).sum()
            length_list.append(seq_len)
        start_index = 0
        for seq_len in length_list:
            y_original.append(labels[start_index: start_index + seq_len])
            start_index = start_index + seq_len
        return y_original


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round(elapsed))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_epoch(model, train_dataloader, optimizer, device, logger, use_crf, loss_fn):
    total_loss_train = 0

    t0 = time.time()
    model.train()
    for step, (attention_masks, labels, bio_features, bert_features) in enumerate(train_dataloader):
        if step % 40 == 0 and step != 0:
            elapsed = format_time(time.time() - t0)
            logger.info(
                '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        bio_features, bert_features, labels, attention_masks = bio_features.to(device), bert_features.to(device), \
                                                               labels.to(device), attention_masks.to(device)

        optimizer.zero_grad()
        model_output = model.forward(bio_features=bio_features,
                                     bert_features=bert_features,
                                     attention_mask=attention_masks,
                                     labels=labels)
        if use_crf:  # 使用crf，则训练时模型的返回值有两个，pre_logits和loss
            y_pred, loss = model_output
        else:
            y_true = get_label_from_mask(labels, attention_masks)
            y_pred = model_output
            loss = loss_fn(y_pred, y_true.type(torch.long))
        loss.backward()
        optimizer.step()

        total_loss_train = total_loss_train + loss.cpu().detach().numpy()

    training_time = format_time(time.time() - t0)
    logger.info("")
    logger.info("  Average training loss: {0:.6f}".format(total_loss_train / len(train_dataloader)))
    logger.info("  Training epcoh took: {:}".format(training_time))

    return total_loss_train / len(train_dataloader)


def validation_epoch(model, validation_dataloader, device, logger, epoch, use_crf, loss_fn, res_csv):
    total_loss_val = 0
    pre_labels, real_labels = [], []
    model.eval()
    with torch.no_grad():
        for attention_masks, labels, bio_features, bert_features in validation_dataloader:
            bio_features, bert_features, labels, attention_masks = bio_features.to(device), bert_features.to(device), \
                                                                   labels.to(device), attention_masks.to(device)
            y_true = get_label_from_mask(labels, attention_masks)
            real_labels.append(y_true)
            model_output = model.forward(bio_features=bio_features,
                                         bert_features=bert_features,
                                         attention_mask=attention_masks,
                                         labels=labels)
            if use_crf:
                y_pred, loss = model_output
                # get validation predict label
                for i, seq_prediction in enumerate(y_pred):  # seq_predictions shape is (max_len, target_size), (512, 2)
                    trans = model.state_dict()["crf_layer.trans"]
                    trans = trans.detach().cpu()
                    seq_prediction = seq_prediction.detach().cpu()
                    viterbi_decode_result = viterbi_decode(seq_prediction.squeeze(0), trans)
                    pre_labels.append(viterbi_decode_result[0: (attention_masks[i] == 1).sum()])
            else:
                y_pred = model_output
                loss = loss_fn(y_pred, y_true.type(torch.long))
                # pre_labels.append(torch.softmax(torch.argmax(y_pred, dim=1), dim=1))
                pre_labels.append(torch.argmax(torch.softmax(y_pred, dim=1), dim=1))

            total_loss_val = total_loss_val + loss

    real_labels, pre_labels = torch.cat(real_labels, 0), torch.cat(pre_labels, 0)
    real_labels = real_labels.detach().cpu().numpy().tolist()
    pre_labels = pre_labels.detach().cpu().numpy().tolist()

    # 将结果写入csv文件
    classification_performance = metrics_classification(np.array(pre_labels), np.array(real_labels))
    res = ["validation", epoch]
    res.extend(classification_performance)
    res_csv.loc[len(res_csv)] = res
    logger.info("  Average validation loss: {0:.6f}".format(total_loss_val / len(validation_dataloader)))
    logger.info("  Epoch:{},performance on validation".format(epoch + 1))
    logger.info("precision:{:.4f}, recall:{:.4f}, accuracy:{:.4f}, F1:{:.4f}, auc:{:.4f}, auprc:{:.4f},"
                "mcc:{:.4f}, "
                "ppv:{:.4f}, npv:{:.4f}, tpr:{:.4f}, tnr:{:.4f} "
                .format(classification_performance[0], classification_performance[1],
                        classification_performance[2], classification_performance[3],
                        classification_performance[4], classification_performance[5],
                        classification_performance[6], classification_performance[7],
                        classification_performance[8], classification_performance[9], classification_performance[10]))
    return total_loss_val / len(validation_dataloader)


def test_epoch(model, test_dataloader_dict, device, logger, use_crf, res_csv, epoch):
    model.eval()
    for test_name, test_dataloader in test_dataloader_dict.items():
        pre_labels, real_labels = [], []
        with torch.no_grad():
            for attention_masks, labels, bio_features, bert_features in test_dataloader:
                bio_features, bert_features, labels, attention_masks = bio_features.to(device), bert_features.to(
                    device), \
                                                                       labels.to(device), attention_masks.to(device)
                real_labels.append(get_label_from_mask(labels, attention_masks))

                # get validation predict label
                model_out = model.predict(bio_features=bio_features,
                                          bert_features=bert_features,
                                          attention_mask=attention_masks)

                if use_crf:
                    trans = model.state_dict()["crf_layer.trans"]
                    for i, seq_prediction in enumerate(
                            model_out):  # seq_predictions shape is (max_len, target_size), (512, 2)
                        trans = trans.detach().cpu()
                        seq_prediction = seq_prediction.detach().cpup()
                        viterbi_decode_result = viterbi_decode(seq_prediction.squeeze(0), trans)
                        pre_labels.append(viterbi_decode_result[0: (attention_masks[i] == 1).sum()])
                else:
                    y_pred = torch.argmax(model_out, dim=1)
                    pre_labels.append(y_pred)

        real_labels, pre_labels = torch.cat(real_labels, 0), torch.cat(pre_labels, 0)
        real_labels = real_labels.detach().cpu().numpy().tolist()
        pre_labels = pre_labels.detach().cpu().numpy().tolist()
        pre_labels, real_labels = np.array(pre_labels), np.array(real_labels)
        classification_performance = metrics_classification(pre_labels, real_labels)
        # 将结果写入csv文件
        res = [test_name, epoch]
        res.extend(classification_performance)
        res_csv.loc[len(res_csv)] = res
        logger.info(f"============Performance on {test_name}==============")
        logger.info("precision:{:.4f}, recall:{:.4f}, accuracy:{:.4f}, F1:{:.4f}, auc:{:.4f}, auprc:{:.4f},"
                    "mcc:{:.4f}, "
                    "ppv:{:.4f}, npv:{:.4f}, tpr:{:.4f}, tnr:{:.4f} "
                    .format(classification_performance[0], classification_performance[1],
                            classification_performance[2], classification_performance[3],
                            classification_performance[4], classification_performance[5],
                            classification_performance[6], classification_performance[7],
                            classification_performance[8], classification_performance[9],
                            classification_performance[10]))


def train_for_single_epoch(model, train_dataloader, optimizer, device, logger, use_crf, loss_fn):
    """
    只对一种特征进行训练（bio or bert）
    :param model:
    :param train_dataloader:
    :param optimizer:
    :param device:
    :param logger:
    :param use_crf:
    :param loss_fn:
    :return:
    """
    total_loss_train = 0
    t0 = time.time()
    model.train()
    for step, (attention_masks, labels, features) in enumerate(train_dataloader):
        if step % 100 == 0 and step != 0:
            elapsed = format_time(time.time() - t0)
            logger.info(
                '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        features, labels, attention_masks = features.to(device), labels.to(device), attention_masks.to(device)

        optimizer.zero_grad()
        model_output = model.forward(features, attention_masks, labels)
        if use_crf:  # 使用crf，则训练时模型的返回值有两个，pre_logits和loss
            y_pred, loss = model_output
        else:
            y_true = get_label_from_mask(labels, attention_masks)
            y_pred = model_output
            y_pred = torch.squeeze(y_pred, dim=1)  # 针对BCE做的处理
            # loss = loss_fn(y_pred, y_true.type(torch.long))
            loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss_train = total_loss_train + loss.cpu().detach().numpy()
    training_time = format_time(time.time() - t0)
    logger.info("")
    logger.info("  Average training loss: {0:.6f}".format(total_loss_train / len(train_dataloader)))
    logger.info("  Training epcoh took: {:}".format(training_time))
    return total_loss_train / len(train_dataloader)


def validation_for_single_epoch(model, validation_dataloader, device, logger, epoch, use_crf, loss_fn, res_csv):
    total_loss_val = 0
    pre_labels, pre_logits, real_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for attention_masks, labels, features in validation_dataloader:
            features, labels, attention_masks = features.to(device), labels.to(device), attention_masks.to(
                device)
            y_true = get_label_from_mask(labels, attention_masks)
            real_labels.append(y_true)
            model_output = model.forward(features, attention_masks, labels)
            if use_crf:
                y_pred, loss = model_output
                # get validation predict label
                for i, seq_prediction in enumerate(y_pred):  # seq_predictions shape is (max_len, target_size), (512, 2)
                    trans = model.state_dict()["crf_layer.trans"]
                    trans = trans.detach().cpu()
                    seq_prediction = seq_prediction.detach().cpu()
                    viterbi_decode_result = viterbi_decode(seq_prediction.squeeze(0), trans)
                    pre_labels.append(viterbi_decode_result[0: (attention_masks[i] == 1).sum()])
            else:
                # y_pred = model_output
                y_pred = torch.squeeze(model_output, dim=1)  # 针对BCE做处理
                # loss = loss_fn(y_pred, y_true.type(torch.long))
                loss = loss_fn(y_pred, y_true)
                # pre_labels.append(torch.argmax(torch.softmax(y_pred, dim=1), dim=1))
                y_pred = torch.nn.functional.sigmoid(y_pred)
                pre_logits.append(y_pred)
                pre_labels.append(torch.where(y_pred >= 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)))

            total_loss_val = total_loss_val + loss

    real_labels, pre_labels, pre_logits = torch.cat(real_labels, 0), torch.cat(pre_labels, 0), torch.cat(pre_logits, 0)
    real_labels = real_labels.detach().cpu().numpy().tolist()
    pre_labels = pre_labels.detach().cpu().numpy().tolist()
    pre_logits = pre_logits.detach().cpu().numpy().tolist()

    # 将结果写入csv文件
    classification_performance = metrics_classification(y_pre_class=np.array(pre_labels), y_label=np.array(real_labels),
                                                        y_pre_logit=np.array(pre_logits))
    res = ["validation", epoch]
    res.extend(classification_performance)
    res_csv.loc[len(res_csv)] = res
    logger.info("  Average validation loss: {0:.6f}".format(total_loss_val / len(validation_dataloader)))
    logger.info("  Epoch:{},performance on validation".format(epoch + 1))
    logger.info("precision:{:.4f}, recall:{:.4f}, accuracy:{:.4f}, F1:{:.4f}, auc:{:.4f}, auprc:{:.4f},"
                "mcc:{:.4f}, "
                "ppv:{:.4f}, npv:{:.4f}, tpr:{:.4f}, tnr:{:.4f} "
                .format(classification_performance[0], classification_performance[1],
                        classification_performance[2], classification_performance[3],
                        classification_performance[4], classification_performance[5],
                        classification_performance[6], classification_performance[7],
                        classification_performance[8], classification_performance[9], classification_performance[10]))
    return total_loss_val / len(validation_dataloader)


def test_for_single_epoch(model, test_dataloader_dict, device, logger, use_crf, res_csv, epoch):
    model.eval()
    auprc = 0
    for test_name, test_dataloader in test_dataloader_dict.items():
        pre_labels, pre_logits, real_labels = [], [], []
        with torch.no_grad():
            for attention_masks, labels, features in test_dataloader:
                features, labels, attention_masks = features.to(device), labels.to(device), attention_masks.to(
                    device)
                real_labels.extend(get_label_list_from_mask(labels, attention_masks, source="real"))

                # get test predict label
                model_out = model.predict(features, attention_masks)

                if use_crf:
                    trans = model.state_dict()["crf_layer.trans"]
                    for i, seq_prediction in enumerate(
                            model_out):  # seq_predictions shape is (max_len, target_size), (512, 2)
                        trans = trans.detach().cpu()
                        seq_prediction = seq_prediction.detach().cpu()
                        viterbi_decode_result = viterbi_decode(seq_prediction.squeeze(0), trans)
                        pre_labels.append(viterbi_decode_result[0: (attention_masks[i] == 1).sum()])
                else:
                    # y_pred = torch.argmax(model_out, dim=1)
                    # pre_labels.append(y_pred)
                    model_out = torch.squeeze(model_out, dim=1)
                    pre_logits.extend(get_label_list_from_mask(model_out, attention_masks, source="predict"))
                    class_labels = torch.where(model_out >= 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
                    pre_labels.extend(get_label_list_from_mask(class_labels, attention_masks, source="predict"))

        performance_list = []
        for i in range(len(pre_labels)):
            performance_list.append(metrics_classification(y_label=real_labels[i], y_pre_class=pre_labels[i],
                                                           y_pre_logit=pre_logits[i]))
        performance_list = np.array(performance_list)
        classification_performance = np.mean(performance_list, axis=0)
        # 将结果写入csv文件
        res = [test_name, epoch]
        res.extend(classification_performance)
        res_csv.loc[len(res_csv)] = res
        logger.info(f"============Performance on {test_name}==============")
        logger.info("precision:{:.4f}, recall:{:.4f}, accuracy:{:.4f}, F1:{:.4f}, auc:{:.4f}, auprc:{:.4f},"
                    "mcc:{:.4f}, "
                    "ppv:{:.4f}, npv:{:.4f}, tpr:{:.4f}, tnr:{:.4f} "
                    .format(classification_performance[0], classification_performance[1],
                            classification_performance[2], classification_performance[3],
                            classification_performance[4], classification_performance[5],
                            classification_performance[6], classification_performance[7],
                            classification_performance[8], classification_performance[9],
                            classification_performance[10]))
        if test_name == "Dset_164":
            auprc = classification_performance[5]
    return auprc


def train_for_multi_epoch(model, train_dataloader, optimizer, device, logger, use_crf, loss_fn):
    """
    针对bio和bert同时存在的场景
    :param model:
    :param train_dataloader:
    :param optimizer:
    :param device:
    :param logger:
    :param use_crf:
    :param loss_fn:
    :return:
    """
    total_loss_train = 0
    t0 = time.time()
    model.train()
    for step, (attention_masks, labels, bio_features, bert_features) in enumerate(train_dataloader):
        if step % 100 == 0 and step != 0:
            elapsed = format_time(time.time() - t0)
            logger.info(
                '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        bio_features, bert_features, labels, attention_masks = bio_features.to(device), bert_features.to(
            device), labels.to(device), attention_masks.to(device)

        optimizer.zero_grad()
        model_output = model.forward(bio_features, bert_features, attention_masks, labels)
        if use_crf:  # 使用crf，则训练时模型的返回值有两个，pre_logits和loss
            y_pred, loss = model_output
        else:
            y_true = get_label_from_mask(labels, attention_masks)
            y_pred = model_output
            y_pred = torch.squeeze(y_pred, dim=1)  # 针对BCE做的处理
            # loss = loss_fn(y_pred, y_true.type(torch.long))
            loss = loss_fn(y_pred, y_true)
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        total_loss_train = total_loss_train + loss.cpu().detach().numpy()
    training_time = format_time(time.time() - t0)
    logger.info("")
    logger.info("  Average training loss: {0:.6f}".format(total_loss_train / len(train_dataloader)))
    logger.info("  Training epcoh took: {:}".format(training_time))
    return total_loss_train / len(train_dataloader)


def validation_for_multi_epoch(model, validation_dataloader, device, logger, epoch, use_crf, loss_fn, res_csv):
    total_loss_val = 0
    pre_labels, pre_logits, real_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for attention_masks, labels, bio_features, bert_features in validation_dataloader:
            bio_features, bert_features, labels, attention_masks = bio_features.to(device), bert_features.to(device), \
                                                                   labels.to(device), attention_masks.to(
                device)
            y_true = get_label_from_mask(labels, attention_masks)
            real_labels.append(y_true)
            model_output = model.forward(bio_features, bert_features, attention_masks, labels)
            if use_crf:
                y_pred, loss = model_output
                # get validation predict label
                for i, seq_prediction in enumerate(y_pred):  # seq_predictions shape is (max_len, target_size), (512, 2)
                    trans = model.state_dict()["crf_layer.trans"]
                    trans = trans.detach().cpu()
                    seq_prediction = seq_prediction.detach().cpu()
                    viterbi_decode_result = viterbi_decode(seq_prediction.squeeze(0), trans)
                    pre_labels.append(viterbi_decode_result[0: (attention_masks[i] == 1).sum()])
            else:
                # y_pred = model_output
                y_pred = torch.squeeze(model_output, dim=1)  # 针对BCE做处理
                # loss = loss_fn(y_pred, y_true.type(torch.long))
                loss = loss_fn(y_pred, y_true)
                # pre_labels.append(torch.argmax(torch.softmax(y_pred, dim=1), dim=1))
                y_pred = torch.nn.functional.sigmoid(y_pred)
                pre_logits.append(y_pred)
                pre_labels.append(torch.where(y_pred >= 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device)))

            total_loss_val = total_loss_val + loss.mean().cpu().detach().numpy()

    real_labels, pre_labels, pre_logits = torch.cat(real_labels, 0), torch.cat(pre_labels, 0), torch.cat(pre_logits, 0)
    real_labels = real_labels.detach().cpu().numpy().tolist()
    pre_labels = pre_labels.detach().cpu().numpy().tolist()
    pre_logits = pre_logits.detach().cpu().numpy().tolist()

    # 将结果写入csv文件
    classification_performance = metrics_classification(y_pre_class=np.array(pre_labels), y_label=np.array(real_labels),
                                                        y_pre_logit=np.array(pre_logits))
    res = ["validation", epoch]
    res.extend(classification_performance)
    res_csv.loc[len(res_csv)] = res
    logger.info("  Average validation loss: {0:.6f}".format(total_loss_val / len(validation_dataloader)))
    logger.info("  Epoch:{},performance on validation".format(epoch + 1))

    return total_loss_val / len(validation_dataloader)


def test_for_multi_epoch(model, test_dataloader_dict, device, logger, use_crf, res_csv, epoch):
    model.eval()
    for test_name, test_dataloader in test_dataloader_dict.items():
        pre_labels, pre_logits, real_labels = [], [], []
        with torch.no_grad():
            for attention_masks, labels, bio_features, bert_features in test_dataloader:
                bio_features, bert_features, labels, attention_masks = bio_features.to(device), bert_features.to(
                    device), \
                                                                       labels.to(device), attention_masks.to(device)
                real_labels.extend(get_label_list_from_mask(labels, attention_masks, source="real"))

                # get test predict label
                model_out = model.predict(bio_features, bert_features, attention_masks)

                if use_crf:
                    trans = model.state_dict()["crf_layer.trans"]
                    for i, seq_prediction in enumerate(
                            model_out):  # seq_predictions shape is (max_len, target_size), (512, 2)
                        trans = trans.detach().cpu()
                        seq_prediction = seq_prediction.detach().cpu()
                        viterbi_decode_result = viterbi_decode(seq_prediction.squeeze(0), trans)
                        pre_labels.append(viterbi_decode_result[0: (attention_masks[i] == 1).sum()])
                else:
                    # y_pred = torch.argmax(model_out, dim=1)
                    # pre_labels.append(y_pred)
                    model_out = torch.squeeze(model_out, dim=1)
                    pre_logits.extend(get_label_list_from_mask(model_out, attention_masks, source="predict"))
                    class_labels = torch.where(model_out >= 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
                    pre_labels.extend(get_label_list_from_mask(class_labels, attention_masks, source="predict"))

        performance_list = []
        for i in range(len(pre_labels)):
            performance_list.append(metrics_classification(y_label=real_labels[i], y_pre_class=pre_labels[i],
                                                           y_pre_logit=pre_logits[i]))
        performance_list = np.array(performance_list)
        classification_performance = np.mean(performance_list, axis=0)
        # 将结果写入csv文件
        res = [test_name, epoch]
        res.extend(classification_performance)
        res_csv.loc[len(res_csv)] = res
        logger.info(f"============Performance on {test_name}==============")
        logger.info("precision:{:.4f}, recall:{:.4f}, accuracy:{:.4f}, F1:{:.4f}, auc:{:.4f}, auprc:{:.4f},"
                    "mcc:{:.4f}, "
                    "ppv:{:.4f}, npv:{:.4f}, tpr:{:.4f}, tnr:{:.4f} "
                    .format(classification_performance[0], classification_performance[1],
                            classification_performance[2], classification_performance[3],
                            classification_performance[4], classification_performance[5],
                            classification_performance[6], classification_performance[7],
                            classification_performance[8], classification_performance[9],
                            classification_performance[10]))
        if test_name == "Dset_186":
            auprc = classification_performance[5]
    return auprc


def train_bert_ft_epoch(model, train_dataloader, optimizer, device, logger, loss_fn):
    total_loss_train = 0

    t0 = time.time()
    model.train()
    for step, (input_ids, labels, attention_masks) in enumerate(train_dataloader):
        if step % 100 == 0 and step != 0:
            elapsed = format_time(time.time() - t0)
            logger.info(
                '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        input_ids, labels, attention_masks = input_ids.to(device), labels.to(device), attention_masks.to(device)

        y_true = []
        for seq_num in range(len(labels)):
            seq_len = (attention_masks[seq_num] == 1).sum()
            label = labels[seq_num][0:seq_len]
            y_true.append(label)

        optimizer.zero_grad()
        y_true = torch.cat(y_true, 0).view(-1, )
        y_pred = model.forward(input_ids=input_ids, attention_mask=attention_masks)
        loss = loss_fn(y_pred, y_true.type(torch.long))
        loss.backward()
        optimizer.step()

        total_loss_train = total_loss_train + loss.cpu().detach().numpy()

    training_time = format_time(time.time() - t0)
    logger.info("")
    logger.info("  Average training loss: {0:.6f}".format(total_loss_train / len(train_dataloader)))
    logger.info("  Training epcoh took: {:}".format(training_time))

    return total_loss_train / len(train_dataloader)


def validation_bert_ft_epoch(model, validation_dataloader, device, logger, loss_fn, epoch):
    softmax_fn = torch.nn.Softmax(dim=1)
    total_loss_val = 0
    pre_labels, real_labels = [], []
    model.eval()
    with torch.no_grad():
        for input_ids, labels, attention_masks in validation_dataloader:
            input_ids, labels, attention_masks = input_ids.to(device), labels.to(device), attention_masks.to(device)
            y_pred = model.forward(input_ids=input_ids, attention_mask=attention_masks)
            y_true = []  # 计算损失用
            for seq_num in range(len(labels)):
                real_labels.append(labels[seq_num][0: (attention_masks[seq_num] == 1).sum()].view(-1, ))
                y_true.append(labels[seq_num][0: (attention_masks[seq_num] == 1).sum()].view(-1, ))

            # 计算损失
            y_true = torch.cat(y_true, 0)
            loss = loss_fn(y_pred, y_true.type(torch.long))
            total_loss_val = total_loss_val + loss

            # 汇总标签，后续计算分类性能指标
            y_pred = softmax_fn(y_pred)
            y_pred = torch.argmax(y_pred, dim=1)
            pre_labels.append(y_pred.view(-1, ))

    real_labels, pre_labels = torch.cat(real_labels, 0), torch.cat(pre_labels, 0)
    real_labels = real_labels.detach().cpu().numpy().tolist()
    pre_labels = pre_labels.detach().cpu().numpy().tolist()
    pre_labels, real_labels = np.array(pre_labels), np.array(real_labels)
    classification_performance = metrics_classification(np.array(pre_labels), np.array(real_labels))
    logger.info("  Average validation loss: {0:.6f}".format(total_loss_val / len(validation_dataloader)))
    logger.info("  Epoch:{},performance on validation".format(epoch + 1))
    logger.info("precision:{:.4f}, recall:{:.4f}, accuracy:{:.4f}, F1:{:.4f}, auc:{:.4f}, auprc:{:.4f},"
                "mcc:{:.4f}, "
                "ppv:{:.4f}, npv:{:.4f}, tpr:{:.4f}, tnr:{:.4f} "
                .format(classification_performance[0], classification_performance[1],
                        classification_performance[2], classification_performance[3],
                        classification_performance[4], classification_performance[5],
                        classification_performance[6], classification_performance[7],
                        classification_performance[8], classification_performance[9], classification_performance[10]))
    return classification_performance[2], total_loss_val / len(validation_dataloader)


def test_bert_ft_epoch(model, test_dataloader, device, logger):
    pre_labels, real_labels = [], []

    model.eval()
    with torch.no_grad():
        for input_ids, labels, attention_masks in test_dataloader:
            input_ids, labels, attention_masks = input_ids.to(device), labels.to(device), attention_masks.to(device)

            for seq_num in range(len(labels)):
                real_labels.append(labels[seq_num][0: (attention_masks[seq_num] == 1).sum()].view(-1, ))

            # get validation predict label
            y_pred = model.predict(input_ids=input_ids, attention_mask=attention_masks)
            pre_labels.append(y_pred.view(-1, ))

    real_labels, pre_labels = torch.cat(real_labels, 0), torch.cat(pre_labels, 0)
    real_labels = real_labels.detach().cpu().numpy().tolist()
    pre_labels = pre_labels.detach().cpu().numpy().tolist()
    pre_labels, real_labels = np.array(pre_labels), np.array(real_labels)

    classification_performance = metrics_classification(pre_labels, real_labels)
    logger.info("============Performance on test==============")
    logger.info("precision:{:.4f}, recall:{:.4f}, accuracy:{:.4f}, F1:{:.4f}, auc:{:.4f}, auprc:{:.4f},"
                "mcc:{:.4f}, "
                "ppv:{:.4f}, npv:{:.4f}, tpr:{:.4f}, tnr:{:.4f} "
                .format(classification_performance[0], classification_performance[1],
                        classification_performance[2], classification_performance[3],
                        classification_performance[4], classification_performance[5],
                        classification_performance[6], classification_performance[7],
                        classification_performance[8], classification_performance[9], classification_performance[10]))
