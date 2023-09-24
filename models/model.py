#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2022/3/11 16:52
# @Author   : Dling

# @FileName : model.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import torch.nn as nn
from models.text_model import *
from models.neighbor_model import *
from models.path_model import *


class HGDModel(nn.Module):
    def __init__(self, config, pretrained_ent_emb, pretrained_rel_emb):
        super().__init__()
        self.sent_out_dim = config['text_out_dim']
        self.neigh_out_dim = config['neigh_output_dim']
        self.path_out_dim = config['path_output_dim']
        self.dropout = config['dropout']
        self.lamd = config['lamd']
        self.device0 = config['device0']
        self.device1 = config['device1']

        concept_num, concept_dim = pretrained_ent_emb.shape[0], pretrained_ent_emb.shape[1]
        relation_num, relation_dim = pretrained_rel_emb.shape[0], pretrained_rel_emb.shape[1]
        self.concept_emb = nn.Embedding(concept_num, concept_dim)
        self.relation_emb = nn.Embedding(relation_num, relation_dim)

        self.concept_emb.weight.data.copy_(torch.from_numpy(pretrained_ent_emb))
        self.relation_emb.weight.data.copy_(torch.from_numpy(pretrained_rel_emb))

        self.text_encoder = TextEncoder(config).to(self.device0)
        self.neighbor_encoder = NeighborEncoder(config, concept_dim, relation_dim).to(self.device1)
        self.path_encoder = PathEncoder(config, concept_dim, relation_dim).to(self.device0)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.sent_out_dim+self.path_out_dim, self.sent_out_dim),
            # nn.Linear(self.sent_out_dim+self.path_out_dim, self.sent_out_dim),
            nn.LeakyReLU(),
            # nn.Dropout(self.dropout),
            nn.Linear(self.sent_out_dim, 2)
        ).to(self.device0)
        self.loss_fc = torch.nn.CrossEntropyLoss().to(self.device0)

    def forward(self, batch, training=True):
        batch_ids, batch_labels, batch_text_data, batch_path_lists, batched_path_graph, concept_mapping_dicts, \
        rel_mapping_dicts, batched_sub1_graph, mask_sg1s, batched_neigh_data, \
        batched_neigh_flag, batched_nsub_graph, batched_nmask_sgs = batch
        ep_text_rep, sen_text_rep, e1_text_rep, e2_text_rep = self.text_encoder(batch_text_data)
        neigh_rep, loss_n = self.neighbor_encoder(self.concept_emb.to(self.device1), self.relation_emb.to(self.device1),
                                          e1_text_rep.to(self.device1), e2_text_rep.to(self.device1), batched_neigh_data,
                                          batched_neigh_flag, batched_nsub_graph, batched_nmask_sgs, training)
        # path_rep, loss_p = self.path_encoder(self.concept_emb.to(self.device0), self.relation_emb.to(self.device0),
        #                              e1_text_rep.to(self.device0), e2_text_rep.to(self.device0), batch_path_lists,
        #                              batched_path_graph, concept_mapping_dicts, rel_mapping_dicts, batched_sub1_graph,
        #                              mask_sg1s, training)
        ep_rep = torch.cat([ep_text_rep, neigh_rep], dim=-1)
        # ep_rep = torch.cat([ep_text_rep, path_rep],  dim=-1)
        logits = self.classifier(ep_rep)
        _, predicted_label = logits.max(1)

        if training:
            loss_c = self.loss_fc(logits, batch_labels)
            loss = loss_c + (1 - self.lamd) * loss_n
            return loss, loss_c, loss_n, loss_n, logits, predicted_label
            # return loss_c, loss_c, loss_c, loss_c, logits, predicted_label
        else:
            return logits, predicted_label
