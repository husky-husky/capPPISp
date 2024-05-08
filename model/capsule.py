# coding: utf-8
# @Author    :陈梦淇
# @time      :2023/11/15
import os
import sys

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        """
        单个胶囊网络层
        :param num_capsules: 胶囊个数
        :param num_route_nodes:
        :param num_iterations: 动态路由迭代次数
        """
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.relu = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm1d(num_capsules)

        if self.num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(self.num_capsules, self.num_route_nodes,
                                                          in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding="same")
                 for _ in range(self.num_capsules)]
            )

    @staticmethod
    def squash(tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        """

        :param x: (batch_size, 1, seq_len, embedding_dim)
        :return:
        """
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size()))
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            # outputs = []
            # for capsule in self.capsules:
            #     c_o = capsule(x)
            #     c_o_transpose = c_o.permute(1, 0, 2)
            #     c_o_flatten = c_o_transpose.reshape(-1)
            #     c_o_reshape = c_o_flatten.view(x.size(1), -1)
            #     outputs.append(c_o_reshape.unsqueeze(-1))
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.relu(self.bn(outputs.permute(0, 2, 1))).permute(0, 2, 1)
            outputs = self.squash(outputs)
        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, input_size, out_num_capsules=2):
        """

        :param input_size: 特征维度
        :param out_num_capsules:
        """
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 3), stride=1, padding="same")
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=1,
                                             out_channels=2, kernel_size=(1, 9), stride=1)

        self.cls_capsules = CapsuleLayer(num_capsules=out_num_capsules, num_route_nodes=512, in_channels=8,
                                         out_channels=16)

        self.relu = torch.nn.ReLU()

    def forward(self, x, y=None):

        # x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        capsule_output = self.cls_capsules(x).squeeze().transpose(0, 1)

        # capsule_output = []
        # for s in x:
        #     s = self.primary_capsules(s)
        #     s = self.cls_capsules(s).squeeze().transpose(0, 1)
        #     capsule_output.append(s.unsqueeze(0))
        # x = self.primary_capsules(x)
        # x = self.cls_capsules(x).squeeze().transpose(0, 1)
        # capsule_output = torch.cat(capsule_output, dim=0)
        classes = (capsule_output ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        return classes


if __name__ == "__main__":
   pass
