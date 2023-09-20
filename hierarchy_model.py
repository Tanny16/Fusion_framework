# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.self_attention import SelfAttention, TitleAwareSelfAttention, TitleAwareGateSelfAttention


class HierarchyModel(nn.Module):
    """
    Hierarchical Attention Model
    """

    def __init__(self, input_size, num_classes):
        super(HierarchyModel, self).__init__()
        self.word_attention_layer = SelfAttention(input_size=input_size)
        self.sentence_attention_layer = SelfAttention(input_size=input_size)
        self.para_attention_layer = SelfAttention(input_size=input_size)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # word attention layer
        x, word_attention_score = self.word_attention_layer(x)
        x = torch.sum(x, dim=-2)
        x = torch.squeeze(x, -2)

        # sentence attention layer
        x, sentence_attention_score = self.sentence_attention_layer(x)
        x = torch.sum(x, dim=-2)
        x = torch.squeeze(x, -2)

        # para attention layer
        x, para_attention_score = self.para_attention_layer(x)
        x = torch.sum(x, dim=-2)
        x = torch.squeeze(x, -2)

        logits = self.fc(x)
        y = F.softmax(logits, dim=-1)
        return y, logits, word_attention_score, sentence_attention_score, para_attention_score


class TitleHierarchyModel(nn.Module):
    """
    Hierarchical Attention Model
    """

    def __init__(self, input_size, num_classes):
        super(TitleHierarchyModel, self).__init__()
        self.word_attention_layer = SelfAttention(input_size=input_size)
        self.sentence_attention_layer = TitleAwareSelfAttention(input_size=input_size)
        self.para_attention_layer = SelfAttention(input_size=input_size)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x, t):
        # word attention layer
        x, word_attention_score = self.word_attention_layer(x)
        x = torch.sum(x, dim=-2)
        x = torch.squeeze(x, -2)

        # sentence attention layer
        x, sentence_attention_score = self.sentence_attention_layer(x, t)
        x = torch.sum(x, dim=-2)
        x = torch.squeeze(x, -2)

        # para attention layer
        x, para_attention_score = self.para_attention_layer(x)
        x = torch.sum(x, dim=-2)
        x = torch.squeeze(x, -2)

        logits = self.fc(x)
        y = F.softmax(logits, dim=-1)
        return y, logits, word_attention_score, sentence_attention_score, para_attention_score


class TitleGateHierarchyModel(nn.Module):
    """
    Hierarchical Attention Model
    """

    def __init__(self, input_size, num_classes):
        super(TitleGateHierarchyModel, self).__init__()
        self.word_attention_layer = SelfAttention(input_size=input_size)
        self.sentence_attention_layer = TitleAwareGateSelfAttention(input_size=input_size)
        self.para_attention_layer = SelfAttention(input_size=input_size)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x, t):
        # word attention layer
        x, word_attention_score = self.word_attention_layer(x)
        x = torch.sum(x, dim=-2)
        x = torch.squeeze(x, -2)

        # sentence attention layer
        x, sentence_attention_score = self.sentence_attention_layer(x, t)
        x = torch.sum(x, dim=-2)
        x = torch.squeeze(x, -2)

        # para attention layer
        x, para_attention_score = self.para_attention_layer(x)
        x = torch.sum(x, dim=-2)
        x = torch.squeeze(x, -2)

        logits = self.fc(x)
        y = F.softmax(logits, dim=-1)
        return y, logits, word_attention_score, sentence_attention_score, para_attention_score