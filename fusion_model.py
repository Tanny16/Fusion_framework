import torch.nn as nn
import torch
import torch.nn.functional as F

from model.hierarchy_model import HierarchyModel


class FusionModel(nn.Module):
    """
    Fusion Model
    """

    def __init__(self, input_size, num_classes, news_vec_size, num_base, alpha=0.5):
        super(FusionModel, self).__init__()
        self.hierarchy_layer = HierarchyModel(input_size=input_size, num_classes=news_vec_size)
        self.coefficient = torch.nn.Parameter(torch.ones(1, num_base))
        # self.expand_fc = nn.Linear(news_vec_size + num_base, num_base * news_vec_size)
        self.alpha = alpha
        self.news_vec_size = news_vec_size
        self.fc = nn.Linear(num_base * 2 + news_vec_size, num_classes)

    def forward(self, news, stock):
        _, news, word_attention_score, sentence_attention_score, para_attention_score = self.hierarchy_layer(news)

        combined = torch.cat((news, stock), -1)

        stock = torch.unsqueeze(stock, -2)
        news = torch.unsqueeze(news, -1)

        # Integration of stock and news
        fusion = torch.matmul(news, stock)
        coefficient = torch.stack([self.coefficient for _ in range(self.news_vec_size)], dim=1)
        weighted_fusion = torch.mul(fusion, coefficient)

        stock_news_fusion = torch.sum(weighted_fusion, 1)
        stock_news_fusion = torch.squeeze(stock_news_fusion, 1)
        stock_news_fusion = F.softmax(stock_news_fusion, dim=-1)

        weighted_fusion = torch.sum(weighted_fusion, dim=-2)
        weighted_fusion = torch.squeeze(weighted_fusion, -2)
        fusion_concat = torch.cat((weighted_fusion, combined), -1)

        # fusion_concat = self.alpha * torch.flatten(weighted_fusion, start_dim=1, end_dim=-1) + (1 - self.alpha) * self.expand_fc(combined)

        logits = self.fc(fusion_concat)
        y = F.softmax(logits, dim=-1)

        return y, logits, stock_news_fusion, para_attention_score, sentence_attention_score, word_attention_score