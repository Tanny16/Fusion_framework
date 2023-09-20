# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch


class SelfAttention(nn.Module):
    """
    Self Attention for the last dimension
    """

    def __init__(self, input_size: int):
        """
        :param input_size: size of the last dimension
        """
        super(SelfAttention, self).__init__()
        self.weight_layer = nn.Linear(input_size, 1)

    def forward(self, x):
        weights = self.weight_layer(x)
        weights = torch.squeeze(weights, -1)
        attention_score = F.softmax(weights, dim=-1)
        out = torch.unsqueeze(attention_score, -1) * x
        return out, attention_score


class TitleAwareSelfAttention(nn.Module):
    """
    Self Attention for the last dimension
    """

    def __init__(self, input_size: int):
        """
        :param input_size: size of the last dimension
        """
        super(TitleAwareSelfAttention, self).__init__()
        self.weight_layer = nn.Linear(input_size, 1)

    def forward(self, x, t):
        self_weights = self.weight_layer(x)

        x_trans = torch.unsqueeze(x, -2)

        t = torch.unsqueeze(t, -2)
        t = torch.unsqueeze(t, -2)
        t = torch.unsqueeze(t, -1)

        t = t.expand(-1, 10, 15, 768, 1)

        title_similarity = torch.matmul(x_trans, t)
        title_similarity = torch.squeeze(title_similarity, -1)

        weights = torch.add(self_weights, title_similarity)
        weights = torch.squeeze(weights, -1)
        attention_score = F.softmax(weights, dim=-1)
        out = torch.unsqueeze(attention_score, -1) * x
        return out, attention_score


class TitleAwareGateSelfAttention(nn.Module):
    """
    Self Attention for the last dimension
    """

    def __init__(self, input_size: int):
        """
        :param input_size: size of the last dimension
        """
        super(TitleAwareGateSelfAttention, self).__init__()
        self.weight_layer = nn.Linear(input_size, 1)
        self.title_weight_layer = nn.Linear(input_size, 1)
        self.forget_gate = nn.Linear(input_size, 1)
        self.input_gate = nn.Linear(input_size, 1)

    def forward(self, x, t):
        self_weights = self.weight_layer(x)

        t = torch.unsqueeze(t, -2)
        t = torch.unsqueeze(t, -2)
        t = t.expand(-1, 10, 15, -1)

        f = self.forget_gate(t)
        i = self.input_gate(t)

        weights = torch.add(self_weights * f, torch.tanh(self.title_weight_layer(t))* i)

        weights = torch.squeeze(weights, -1)
        attention_score = F.softmax(weights, dim=-1)

        out = torch.unsqueeze(attention_score, -1) * x
        return out, attention_score