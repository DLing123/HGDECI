#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2022/3/11 17:10
# @Author   : Dling
# @FileName : text_model.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import torch
import torch.nn as nn
from transformers.models.bert import BertModel
from utils import *


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sent_hidden_dim = config['text_hidden_dim']
        self.sent_out_dim = config['text_out_dim']
        self.dropout = config['dropout']

        self.bert = BertModel.from_pretrained(config['pretrained_model_path'])
        self.trans_fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(3 * self.sent_hidden_dim, self.sent_hidden_dim),
            nn.LeakyReLU()
            # nn.Linear(self.sent_hidden_dim, self.sent_out_dim)
        )
        self.trans_fc.apply(weight_init)

    def forward(self, batched_text_data):
        sentences, mask_sentences, events1, mask_events1, events2, mask_events2 = batched_text_data
        sequence_output, cls_output = self.bert(sentences, attention_mask=mask_sentences, return_dict=False)
        event1 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(sequence_output, events1)])
        event2 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(sequence_output, events2)])

        m1 = mask_events1.unsqueeze(-1).expand_as(event1).float()  # bs * event_len * dim
        m2 = mask_events2.unsqueeze(-1).expand_as(event2).float()

        event1 = event1 * m1
        event2 = event2 * m2

        # use max_pooling
        # max_pooling = nn.MaxPool1d(event1.shape[1])
        # e1_rep = max_pooling(event1.transpose(1, 2)).squeeze(2)  # bs*dim*event_len-->bs*dim
        # e2_rep = max_pooling(event2.transpose(1, 2)).squeeze(2)
        e1_rep = torch.sum(event1, dim=1)  # bs * dim
        e2_rep = torch.sum(event2, dim=1)  # bs * dim

        event_rep = torch.cat((cls_output, e1_rep, e2_rep), dim=-1)  # bs * dim
        event_rep = self.trans_fc(event_rep)
        return event_rep, cls_output, e1_rep, e2_rep
        # return cls_output, e1_rep, e2_rep
