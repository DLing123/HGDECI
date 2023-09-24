#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2022/3/12 11:39
# @Author   : Dling
# @FileName : rgat.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from tqdm import tqdm
import networkx as nx
import torch
import dgl
import json
import numpy as np
from utils import *


class RelGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_feats, use_nei_attn=True, use_rel=True):
        super(RelGATLayer, self).__init__()
        in_feats = in_feats if use_rel else in_feats - edge_feats
        self.fc_node = nn.Linear(in_feats, out_feats, bias=False)
        self.attn_fc = nn.Linear(out_feats, 1, bias=False)
        self.fc_edge = nn.Linear(edge_feats, edge_feats, bias=False)

        self.use_attn = use_nei_attn
        self.use_rel = use_rel

        self.node_trans = nn.Linear(2*out_feats, out_feats)
        self.edge_trans = nn.Linear(2*edge_feats, edge_feats)

        self.fc_node.apply(weight_init)
        self.attn_fc.apply(weight_init)
        self.fc_edge.apply(weight_init)
        self.node_trans.apply(weight_init)
        self.edge_trans.apply(weight_init)

    def edge_attention(self, edges):
        src_h, dst_h, rel_h, rel_id = edges.src['node_h'], edges.dst['node_h'], edges.data['rel_h'], edges.data['cncpt_rel']
        all_list = []
        for s_h, d_h, r_h, r in zip(src_h, dst_h, rel_h, rel_id):
            if self.use_rel:
                if r.item() >= 17:
                    all_list.append(torch.cat([d_h, s_h, r_h], dim=-1).unsqueeze(0))
                else:
                    all_list.append(torch.cat([s_h, d_h, r_h], dim=-1).unsqueeze(0))
            else:
                if r.item() >= 17:
                    all_list.append(torch.cat([d_h, s_h], dim=-1).unsqueeze(0))
                else:
                    all_list.append(torch.cat([s_h, d_h], dim=-1).unsqueeze(0))
        all_feats = self.fc_node(torch.cat(all_list, dim=0))  # num*dim
        scores = F.leaky_relu(self.attn_fc(all_feats))
        return {'att_feats': all_feats, 'scores': scores}

    def message_func(self, edges):
        return {'att_h': edges.data['att_feats'], 'att_s': edges.data['scores']}

    def reduce_func(self, nodes):
        if self.use_attn:
            alpha = F.softmax(nodes.mailbox['att_s'], dim=1)
            h = torch.sum(alpha * nodes.mailbox['att_h'], dim=1)
        else:
            h = torch.mean(nodes.mailbox['att_h'], dim=1)
        return {'node_h': h}

    def apply_mod(self, edge):
        rel_h = F.leaky_relu(self.fc_edge(edge.data['rel_h']))
        return {'rel_h': rel_h}

    def forward(self, g, ent_feats, rel_feats):
        g.ndata['node_h'] = ent_feats
        g.edata['rel_h'] = rel_feats
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        # g.apply_edges(func=self.apply_mod)
        g.ndata['node_h'] = F.leaky_relu(self.node_trans(torch.cat([ent_feats, g.ndata['node_h']], dim=-1)))
        # g.edata['rel_h'] = F.leaky_relu(self.edge_trans(torch.cat([rel_feats, g.edata['rel_h']], dim=-1)))
        return g.ndata.pop('node_h'), g.edata.pop('rel_h')


class RGATEncoder(nn.Module):
    def __init__(self, hidden_dim, concept_dim, rel_dim, use_nei_attn=True, use_rel=True, layer=1):
        super(RGATEncoder, self).__init__()
        self.layer = layer

        self.gcn1 = RelGATLayer(2*concept_dim+rel_dim, hidden_dim, rel_dim, use_nei_attn, use_rel)
        if self.layer == 2:
            self.gcn2 = RelGATLayer(2 * hidden_dim + rel_dim, hidden_dim, rel_dim, use_nei_attn, use_rel)

    def forward(self, concept_emb, relation_emb, g):
        ent_feats = concept_emb(g.ndata["cncpt_ids"])
        g_edata = torch.tensor([i - 17 if i >= 17 else i for i in g.edata['cncpt_rel']], device=g.device)
        rel_feats = relation_emb(g_edata)
        node_f, edge_f = self.gcn1(g, ent_feats, rel_feats)
        if self.layer == 2:
            node_f, edge_f = self.gcn2(g, node_f, edge_f)
        g.ndata['node_h'] = node_f
        g.edata['rel_h'] = edge_f
        return g


# def load_graphs():
#     dgs = []
#     with open('../data/story/neighbors.graph1.limited_6.jsonl', 'r', encoding='utf-8') as fin:
#         nxgs = [json.loads(line) for line in fin]
#         nxgs = nxgs[:1]
#     for nxg_doc in tqdm(nxgs, total=len(nxgs), desc='loading graphs'):
#         for sen in nxg_doc['event_pair']:
#             for ep in sen:
#                 if ep['e1_nbs_graph'] is not None:
#                     nxg = nx.node_link_graph(ep['e1_nbs_graph'])
#                     dg = dgl.from_networkx(nxg)
#
#                     cids = [nxg.nodes[n_id]['cid'] for n_id in range(len(dg))]
#                     dg.ndata.update({'cncpt_ids': torch.tensor(cids)})
#                     rels = [nxg.edges[u, v, k]['rel'] for u, v, k in nxg.edges]
#                     dg.edata.update({'cncpt_rel': torch.tensor(rels)})
#                     for u, v, k in nxg.edges:
#                         assert abs(nxg.edges[u, v, k]['rel'] - nxg.edges[v, u, k]['rel']) == 17
#                     dgs.append(dg)
#                 if ep['e2_nbs_graph'] is not None:
#                     nxg = nx.node_link_graph(ep['e2_nbs_graph'])
#                     dg = dgl.from_networkx(nxg)
#
#                     cids = [nxg.nodes[n_id]['cid'] for n_id in range(len(dg))]
#                     dg.ndata.update({'cncpt_ids': torch.tensor(cids)})
#                     rels = [nxg.edges[u, v, k]['rel'] for u, v, k in nxg.edges]
#                     dg.edata.update({'cncpt_rel': torch.tensor(rels)})
#                     for u, v, k in nxg.edges:
#                         assert abs(nxg.edges[u, v, k]['rel'] - nxg.edges[v, u, k]['rel']) == 17
#                     dgs.append(dg)
#                 if len(dgs) > 2:
#                         break
#             break
#     return dgs


# dgs = load_graphs()
# print(dgs[0])
# print(dgs[1])
# dg = dgl.batch(dgs[0:2])
# print(dg)
# print(dgs[0].nodes())
# print(dgs[0].edges())
# print(dgs[0].ndata['cncpt_ids'])
# print(dgs[0].edata['cncpt_rel'])
# print(dgs[1].nodes())
# print(dgs[1].edges())
# print(dgs[1].ndata['cncpt_ids'])
# print(dgs[1].edata['cncpt_rel'])
# print()
# print(dg.nodes())
# print(dg.edges())
# print(dg.ndata['cncpt_ids'])
# print(dg.edata['cncpt_rel'])
#
# pretrained_ent_emb = np.load('../data/transe/glove.transe.sgd.ent.npy')
# pretrained_rel_emb = np.load('../data/transe/glove.transe.sgd.rel.npy')
#
# concept_num, concept_dim = pretrained_ent_emb.shape[0], pretrained_ent_emb.shape[1]
# relation_num, relation_dim = pretrained_rel_emb.shape[0], pretrained_rel_emb.shape[1]
# concept_emb = nn.Embedding(concept_num, concept_dim)
# relation_emb = nn.Embedding(relation_num, relation_dim)
#
# concept_emb.weight.data.copy_(torch.from_numpy(pretrained_ent_emb))
# relation_emb.weight.data.copy_(torch.from_numpy(pretrained_rel_emb))
# gencoder = RGATEncoder(100, concept_dim, relation_dim)
# bgss = dgl.unbatch(gencoder(concept_emb, relation_emb, dg))

# e1_reps = [bgss[0].ndata["node_h"][0]]
# e2_reps = [bgss[1].ndata["node_h"][0]]

# batched_e1_out = torch.stack(e1_reps)
# batched_e2_out = torch.stack(e2_reps)
# batched_neigh_rep = torch.cat([batched_e1_out, batched_e2_out], dim=-1)
# hidden2output = nn.Linear(200, 2)
# batched_neigh_rep = hidden2output(batched_neigh_rep)
# loss_fc = nn.CrossEntropyLoss()
# loss = loss_fc(batched_neigh_rep, torch.tensor([1, 0]))
# loss.backward()
# print(batched_neigh_rep.shape)
