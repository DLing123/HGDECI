#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2022/3/11 22:55
# @Author   : Dling
# @FileName : neighbor_model.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import torch.nn as nn
from models.rgat import *
import torch
from utils import *


class NeighborEncoder(nn.Module):
    def __init__(self, config, concept_dim, relation_dim):
        super(NeighborEncoder, self).__init__()
        self.graph_hidden_dim = config['graph_hidden_dim']
        self.sent_dim = config['text_hidden_dim']
        self.neigh_output_dim = config['neigh_output_dim']
        self.dropout = config['dropout']
        self.tau = config['tau']

        self.concept_dim = concept_dim
        self.relation_dim = relation_dim

        self.trans_fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.graph_hidden_dim, self.neigh_output_dim),
            nn.LeakyReLU()
            # nn.Linear(self.sent_hidden_dim, self.sent_out_dim)
        )
        self.trans_fc.apply(weight_init)
        self.graph_encoder = RGATEncoder(self.graph_hidden_dim, self.concept_dim, self.relation_dim,
                                         config['use_nei_attn'], config['use_rel'])

    def semi_loss(self, z1, z2):
        def pair_sim(z1, z2):
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(pair_sim(z1, z1))  # 4 * 4
        between_sim = f(pair_sim(z1, z2))  # 4 * 4

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def cal_cst_loss(self, concept_emb, relation_emb, bgs, nei_subs1, sub1_masks):
        c1_output_graphs = self.graph_encoder(concept_emb, relation_emb, nei_subs1)
        bgs_sub1 = dgl.unbatch(c1_output_graphs)
        cst_loss = []
        for bg, bg_c1, sub1_mask in zip(bgs, bgs_sub1, sub1_masks):
            bg_rep = bg.ndata["node_h"]
            sub1_emb = bg_c1.ndata["node_h"]
            sub1_rep = bg_rep.new_zeros(bg_rep.size())
            sub1_rep[sub1_mask.nonzero().view(-1)] = sub1_emb
            sub1_index = (sub1_mask == 0).nonzero().view(-1)
            sub1_rep[sub1_index] = concept_emb(bg.ndata["cncpt_ids"][sub1_index])

            cst_loss.append(torch.mean(self.semi_loss(bg_rep, sub1_rep)).unsqueeze(0))
        losses = torch.mean(torch.cat(cst_loss))
        return losses

    def forward(self, concept_emb, relation_emb, e1_vecs, e2_vecs, nei_graphs, graph_flgs, nei_sub1,
                sub1_mask, training=True):
        neigh_output_graphs = self.graph_encoder(concept_emb, relation_emb, nei_graphs)
        bgs = dgl.unbatch(neigh_output_graphs)
        e1_reps, e2_reps = [], []
        index = 0
        for f, e1_vec, e2_vec in zip(graph_flgs, e1_vecs, e2_vecs):
            if f == [0, 0]:
                # e1_neigh_rep = torch.cat([e1_vec, e1_vec.new_zeros(self.graph_hidden_dim)], dim=-1)
                # e2_neigh_rep = torch.cat([e2_vec, e2_vec.new_zeros(self.graph_hidden_dim)], dim=-1)
                e1_neigh_rep = e1_vec.new_zeros(self.graph_hidden_dim)
                e2_neigh_rep = e2_vec.new_zeros(self.graph_hidden_dim)
            else:
                e1_neigh_rep = bgs[index].ndata["node_h"][0]
                e2_neigh_rep = bgs[index+1].ndata["node_h"][0]
                # e1_neigh_rep = torch.cat([e1_vec, e1_graph_emb], dim=-1)
                # e2_neigh_rep = torch.cat([e2_vec, e2_graph_emb], dim=-1)
                index += 2
            e1_reps.append(e1_neigh_rep)
            e2_reps.append(e2_neigh_rep)
        batched_e1_out = torch.stack(e1_reps)
        batched_e2_out = torch.stack(e2_reps)
        # batched_e1_out = self.trans_fc(batched_e1_out)
        # batched_e2_out = self.trans_fc(batched_e2_out)
        batched_neigh_rep = torch.cat([batched_e1_out, batched_e2_out], dim=-1)
        batched_neigh_rep = self.trans_fc(batched_neigh_rep)
        if training:
            losses = self.cal_cst_loss(concept_emb, relation_emb, bgs, nei_sub1, sub1_mask)
            return batched_neigh_rep, losses
        else:
            return batched_neigh_rep, None
