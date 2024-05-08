# coding: utf-8
# @Author    :陈梦淇
# @time      :2023/12/13
import json
import os
import sys

import torch
from torch import nn
from torch.nn import init

from transformers import BertModel, BertTokenizer

current_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
sys.path.append(parent_directory)
from capsule import CapsuleNet


class BiLSTMNet(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden1_size=128,
                 hidden2_size=256,
                 dropout=0.5):
        super(BiLSTMNet, self).__init__()
        self.input_size = input_size
        self.hidden1_size, self.hidden2_size = hidden1_size, hidden2_size
        self.dropout = dropout

        self.layerNorm = nn.LayerNorm(self.input_size)
        self.lstm_layer1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden1_size, batch_first=True,
                                   bidirectional=True, dropout=self.dropout)
        self.lstm_layer2 = nn.LSTM(input_size=self.hidden1_size * 2, hidden_size=self.hidden2_size, batch_first=True,
                                   bidirectional=True, dropout=self.dropout)

        self.output = torch.nn.Linear(self.hidden2_size * 2, output_size)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input_features, attention_mask=None, labels=None):
        x = self.layerNorm(input_features)
        lstm1, _ = self.lstm_layer1(x)
        lstm2, _ = self.lstm_layer2(lstm1)

        if attention_mask is not None:
            # 取出有效部分
            embedding = []
            # 对lstm的输出进行降维到两维
            for seq_num in range(len(attention_mask)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = lstm2[seq_num][0: seq_len]
                embedding.append(seq_emd)
            lstm_o = torch.cat(embedding, 0)
            output = self.output(lstm_o)
            return output
        return lstm2

    def predict(self, input_features, attention_mask=None, labels=None):
        return self.forward(input_features, attention_mask)



class TextCNN(nn.Module):
    def __init__(self, input_size, output_size, pool_size, output_channel=1, dropout=0.4):
        super(TextCNN, self).__init__()
        self.output_channel = output_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, out_channels=self.output_channel, kernel_size=(3, input_size), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((1, pool_size))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, out_channels=self.output_channel, kernel_size=(5, input_size), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((1, pool_size))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, out_channels=self.output_channel, kernel_size=(7, input_size), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((1, pool_size))
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, out_channels=self.output_channel, kernel_size=(9, input_size), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((1, pool_size))
        )

        self.fc = nn.Linear(in_features=int((input_size / 8) * self.output_channel * 4), out_features=output_size)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None):
        """
        :param x: (batch_size, sequence_length, embedding_size)
        :param attention_mask: (batch_size, sequence_length)
        :return:
        """
        x = x.unsqueeze(1)

        conv1_o, conv2_o, conv3_o, conv4_o = self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)
        # 将各个输出通道的结果进行拼接
        conv1_o = [conv1_o[:, i, :, :] for i in range(self.output_channel)]
        conv2_o = [conv2_o[:, i, :, :] for i in range(self.output_channel)]
        conv3_o = [conv3_o[:, i, :, :] for i in range(self.output_channel)]
        conv4_o = [conv4_o[:, i, :, :] for i in range(self.output_channel)]
        conv1_o, conv2_o, conv3_o, conv4_o = torch.cat(conv1_o, dim=-1), torch.cat(conv2_o, dim=-1), \
                                             torch.cat(conv3_o, dim=-1), torch.cat(conv4_o, dim=-1)
        conv_o = torch.cat([conv1_o, conv2_o, conv3_o, conv4_o], dim=-1)  # 将不同卷积的结果进行横向拼接

        if attention_mask is not None:
            # 取出有效部分
            embedding = []
            # 对lstm的输出进行降维到两维
            for seq_num in range(len(attention_mask)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = conv_o[seq_num][0: seq_len]
                embedding.append(seq_emd)
            conv_o = torch.cat(embedding, 0)
            output = self.fc(conv_o)
            return output
        return conv_o

    def predict(self, x, attention_mask=None):
        return self.forward(x, attention_mask)


class TransformerNet(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim=1024, num_heads=8, num_layers=6, dropout_rate=.2):
        super(TransformerNet, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                              dim_feedforward=embedding_dim * 4, dropout=dropout_rate)
        self.transformer_encoder_layers = nn.TransformerEncoder(self.transformer_encoder, num_layers=num_layers)

        self.output_linear = nn.Linear(embedding_dim, output_size)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x, attention_mask=None):
        encoded_output = self.transformer_encoder_layers(x)
        if attention_mask is not None:
            # 取出有效部分
            embedding = []
            # 对lstm的输出进行降维到两维
            for seq_num in range(len(attention_mask)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = encoded_output[seq_num][0: seq_len]
                embedding.append(seq_emd)
            pooled_output = torch.cat(embedding, 0)
            output = self.output(pooled_output)
            return output
        return encoded_output


class EnsembleModel(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size1=128,
                 hidden_size2=256,
                 hidden_size3=512,
                 residual_mode="one", binding_mode="sum"):
        """
        :param residual_mode: none:表示不使用残差，每个模块的输出维度为256
                              one：表示使用残差并使用mean or sum or weight_sum的结合方式
                              two：表示使用残差并使用concat的结合方式，每个模块的输出维度为256
        :param binding_mode:
        """
        super(EnsembleModel, self).__init__()
        self.residual_mode = residual_mode
        self.binding_mode = binding_mode
        self.hidden_size3 = hidden_size3
        self.input_size = input_size

        if self.residual_mode == "none" or self.residual_mode == "two":
            f = open("config_256.json", "r")
        elif self.residual_mode == "one":
            f = open("config_1024.json", "r")
        self.config = json.load(f)
        f.close()

        if self.binding_mode == "weight_sum":
            self.w = torch.nn.Parameter(torch.Tensor([1.0, 1.0, 1.0, 1.0]), requires_grad=True)
            with torch.no_grad():
                self.w.data = nn.functional.softmax(self.w, dim=0).data
        # 将生物特征映射到与bert相同的维度
        self.input_linear = torch.nn.Linear(64, 1024)

        self.bilstm_model = BiLSTMNet(input_size, output_size, hidden1_size=self.config["lstm"]["hidden1"],
                                      hidden2_size=self.config["lstm"]["hidden2"])
        self.bigru_model = BiGRUNet(input_size, output_size, hidden1_size=self.config["gru"]["hidden1"],
                                    hidden2_size=self.config["gru"]["hidden2"])
        self.cnn_model = TextCNN(input_size, output_size, pool_size=self.config["cnn"]["pool_size"],
                                 output_channel=self.config["cnn"]["output_channel"])
        self.transformer_model = TransformerNet(input_size, self.config["transformer"], embedding_dim=input_size)

        self.bilstm = nn.LSTM(input_size=784, hidden_size=hidden_size3, batch_first=True,
                              bidirectional=True, dropout=.4)
        self.output = nn.Linear(in_features=self.hidden_size3, out_features=1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, bio_features, attention_mask, labels=None):
        # 输入预处理
        # bio_features = self.tanh(self.input_linear(bio_features))
        # x = bio_features + bert_features
        x = bio_features

        # 分别经过多个子模块
        lstm_o, gru_o, cnn_o, transformer_o = self.bilstm_model(x), self.bigru_model(x), \
                                              self.cnn_model(x), self.transformer_model(x)
        fea_con = None
        if self.residual_mode == "one":
            lstm_o, gru_o, cnn_o, transformer_o = lstm_o + x, gru_o + x, cnn_o + x, transformer_o + x
            if self.binding_mode == "weight_sum":
                with torch.no_grad():
                    self.w.data = nn.functional.softmax(self.w, dim=0).data
                fea_con = self.w[:, None, None] * torch.stack([lstm_o, gru_o, cnn_o, transformer_o], dim=0)
                fea_con = torch.sum(fea_con, dim=0)
            elif self.binding_mode == "sum":
                fea_con = lstm_o + gru_o + cnn_o + transformer_o
            elif self.binding_mode == "mean":
                fea_con = (lstm_o + gru_o + cnn_o + transformer_o) / 4
        elif self.residual_mode == "two":
            fea_con = torch.cat([lstm_o, gru_o, cnn_o, transformer_o], dim=-1)
            fea_con = fea_con + x
        elif self.residual_mode == "none":
            fea_con = torch.cat([lstm_o, gru_o, cnn_o, transformer_o], dim=-1)

        bilstm_output, _ = self.bilstm(fea_con)
        bilstm_output = bilstm_output[:, :, 0: self.hidden_size3] + bilstm_output[:, :, self.hidden_size3:]

        embedding = []
        # 对lstm的输出进行降维到两维
        for seq_num in range(len(attention_mask)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = bilstm_output[seq_num][0: seq_len]
            embedding.append(seq_emd)
        bilstm_output = torch.cat(embedding, 0)
        output = self.output(bilstm_output)
        return output

    def predict(self, bio_features, attention_mask):
        return self.sigmoid(self.forward(bio_features, attention_mask))



if __name__ == "__main__":
    pass
