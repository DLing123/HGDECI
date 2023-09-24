#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2022/3/11 22:55
# @Author   : Dling
# @FileName : path_model.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from models.rgat import *
from utils import *


class PathEncoder(nn.Module):
    def __init__(self, config, concept_dim, relation_dim):
        super(PathEncoder, self).__init__()
        self.graph_hidden_dim = config['graph_hidden_dim']
        self.lstm_dim = config['lstm_dim']
        self.sent_dim = config['text_hidden_dim']
        self.path_output_dim = config["path_output_dim"]

        self.concept_dim = concept_dim
        self.relation_dim = relation_dim
        bidirect = config['bidirect']
        self.dropout = config['dropout']
        self.tau = config['tau']

        self.use_graph = config['use_graph']
        self.use_lstm = config['use_lstm']
        self.path_attention = config['path_attention']

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=self.relation_dim + self.graph_hidden_dim,
                                hidden_size=self.lstm_dim,
                                bidirectional=bidirect,
                                dropout=self.dropout,
                                batch_first=True)
        else:
            self.lstm_trans = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.relation_dim + self.graph_hidden_dim, self.lstm_dim * 2),  # binary classification
            nn.LeakyReLU()
            )
            self.lstm_trans.apply(weight_init)

        if bidirect:
            self.lstm_dim = self.lstm_dim * 2

        self.hts_encoder = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.graph_hidden_dim, self.lstm_dim),  # binary classification
            nn.LeakyReLU()
        )
        # if self.path_attention:
        #     self.hts_pathlstm_att = nn.Linear(self.ht_encoded_dim, self.lstm_dim)  # transform qas vector to query vectors
        #     self.hts_pathlstm_att.apply(weight_init)

        self.trans_fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.lstm_dim, self.path_output_dim),
            nn.LeakyReLU()
            # nn.Linear(self.sent_hidden_dim, self.sent_out_dim)
        )

        self.lstm.apply(weight_init)
        self.hts_encoder.apply(weight_init)
        self.trans_fc.apply(weight_init)
        self.graph_encoder = RGATEncoder(self.graph_hidden_dim, self.concept_dim, self.relation_dim, layer=2)

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

    def cal_cst_loss(self, concept_emb, relation_emb, path_bg, path_subs1, sub1_masks):
        c1_output_graphs = self.graph_encoder(concept_emb, relation_emb, path_subs1)
        # c2_output_graphs = self.graph_encoder(concept_emb, relation_emb, path_subs2)
        bgs = dgl.unbatch(path_bg)
        bgs_sub1 = dgl.unbatch(c1_output_graphs)
        # bgs_sub2 = dgl.unbatch(c2_output_graphs)
        cst_loss1 = []
        for bg, bg_c1, sub1_mask in zip(bgs, bgs_sub1, sub1_masks):
            bg_rep = bg.ndata["node_h"]
            sub1_emb = bg_c1.ndata["node_h"]
            # sub2_emb = bg_c2.ndata["node_h"]
            sub1_rep = bg_rep.new_zeros(bg_rep.size())
            # sub2_rep = bg_rep.new_zeros(bg_rep.size())
            sub1_rep[sub1_mask.nonzero().view(-1)] = sub1_emb
            sub1_index = (sub1_mask == 0).nonzero().view(-1)
            sub1_rep[sub1_index] = concept_emb(bg.ndata["cncpt_ids"][sub1_index])
            # sub2_rep[sub2_mask.nonzero().view(-1)] = sub2_emb
            # sub2_index = (sub2_mask == 0).nonzero().view(-1)
            # sub2_rep[sub2_index] = concept_emb(bg.ndata["cncpt_ids"][sub2_index])

            cst_loss1.append(torch.mean(self.semi_loss(bg_rep, sub1_rep)).unsqueeze(0))
            # cst_loss2.append(torch.mean(self.semi_loss(bg_rep, sub2_rep)).unsqueeze(0))

        # losses = torch.mean(torch.cat(cst_loss1)) + torch.mean(torch.cat(cst_loss2))
        losses = torch.mean(torch.cat(cst_loss1))
        return losses

    def forward(self, concept_emb, relation_emb, e1_vec_batched, e2_vec_batched, batch_path_lists, graphs,
                concept_mapping_dicts, rel_mapping_dicts, path_sub1, sub1_mask, training=True):
        ht_pairs_batched, cpt_paths_batched, rel_paths_batched, path_len_batched, ht_path_num_batched = batch_path_lists
        # ht_pair_batched: [ [[h,t]](tensor), [[h,t]],  [zero], [[h, t]] ]
        n_ht_pairs = [len(t) for t in ht_pairs_batched]  # num of qa_pairs for each example [1, 1, 0, 1, 0]
        total_ht_pairs = sum(n_ht_pairs)  # total num of qa_pairs
        e1_vec_expanded = e1_vec_batched.new_zeros((total_ht_pairs, e1_vec_batched.size(1)))  # example * dim
        e2_vec_expanded = e2_vec_batched.new_zeros((total_ht_pairs, e2_vec_batched.size(1)))  # example * dim
        i = 0
        for n, e1_vec, e2_vec in zip(n_ht_pairs, e1_vec_batched, e2_vec_batched):
            j = i + n
            e1_vec_expanded[i:j] = e1_vec  # total_ht_pair * dim
            e2_vec_expanded[i:j] = e2_vec
            i = j
        if self.use_graph:
            output_graphs = self.graph_encoder(concept_emb, relation_emb, graphs)
            new_concept_embed = torch.cat((output_graphs.ndata["node_h"], e1_vec_batched.new_zeros((1, self.graph_hidden_dim))))
            new_rel_embed = torch.cat((output_graphs.edata["rel_h"], e1_vec_batched.new_zeros((1, self.relation_dim))))
            new_ht_ids = []  # [[h, t], [h, t], [h, t]]
            mdicted_cpaths = []  # [[[p1], [p2], [p3]], [[p1], [p2], [p3]]]
            mdicted_rels = []
            for ht_ids, cpt_path, mdict, rel_path, rdict in zip(ht_pairs_batched, cpt_paths_batched, concept_mapping_dicts, rel_paths_batched, rel_mapping_dicts):
                # [[h, t]], [[p1], [p2], [p3]],
                if len(mdict) == 0:
                    assert len(mdict) == len(rdict) == len(ht_ids) == len(cpt_path) == len(rel_path)
                id_mapping = lambda x: mdict.get(x, -1)
                new_ht_ids += [[id_mapping(q.item()), id_mapping(a.item())] for q, a in ht_ids]
                new_p = []
                for p in cpt_path:
                    new_p.append([id_mapping(c.item()) for c in p])
                mdicted_cpaths.append(torch.tensor(new_p, dtype=torch.long, device=e1_vec_batched.device))
                rel_mapping = lambda x: rdict.get(x, -1)
                new_r = []
                for p in cpt_path:
                    new_r.append([rel_mapping((p[i].item(), p[i+1].item())) for i in range(len(p)-1)] + [len(new_rel_embed) - 1])
                mdicted_rels.append(torch.tensor(new_r, dtype=torch.long, device=e1_vec_batched.device))

            new_ht_ids = torch.tensor(new_ht_ids, device=e1_vec_batched.device)  # total pairs * 2
            new_ht_vecs = new_concept_embed[new_ht_ids].view(total_ht_pairs, -1)  # total pairs * （2*dim）
            flat_cpt_paths_batched = torch.cat(mdicted_cpaths, 0).to(torch.long)  # path nums * path len
            new_batched_all_ht_cpt_paths_embeds = new_concept_embed[flat_cpt_paths_batched]  # p_nums * p_len * dim
            flat_rel_paths_batched = torch.cat(mdicted_rels, 0).to(torch.long)
            flat_rel_paths_batched = [(i - 17) if i >= 17 else i for i in flat_rel_paths_batched.view(-1)]
            flat_rel_paths_batched = torch.stack(flat_rel_paths_batched, 0).view(-1, 5)
            new_batched_all_ht_rel_paths_embeds = new_rel_embed[flat_rel_paths_batched]
            batched_all_ht_cpt_rel_paths_embeds = torch.cat((new_batched_all_ht_cpt_paths_embeds,
                                                             new_batched_all_ht_rel_paths_embeds), 2)
            if training:
                losses = self.cal_cst_loss(concept_emb, relation_emb, output_graphs, path_sub1, sub1_mask)
        else:
            ht_ids_batched = torch.cat(ht_pairs_batched, 0)  # N x 2
            new_ht_vecs = concept_emb(ht_ids_batched).view(total_ht_pairs, -1)  # qa_pair*(dim*2)
            old_batched_all_ht_cpt_paths_embeds = concept_emb(torch.cat(cpt_paths_batched, 0))  # old concept embed
            pr_temp = []
            for prs in rel_paths_batched:
                for pr in prs:
                    pr_temp.append(torch.stack([i - 17 if i >= 17 else i for i in pr], 0))
            old_batched_all_ht_rel_paths_embeds = relation_emb(torch.stack(pr_temp, 0))
            batched_all_ht_cpt_rel_paths_embeds = torch.cat((old_batched_all_ht_cpt_paths_embeds,
                                                            old_batched_all_ht_rel_paths_embeds), 2)  # path_num * path_len * dim

        # raw_hts_vecs = torch.cat((e1_vec_expanded, e2_vec_expanded, new_ht_vecs), dim=1)  # all the qas triple vectors associated with a statement
        hts_vecs_batched = self.hts_encoder(new_ht_vecs)  # Tij ht_encoder_dim
        # if self.path_attention:
            # query_vecs_batched = self.hts_pathlstm_att(hts_vecs_batched)  # total_ht_pairs * lstm_dim
        if self.use_lstm:
            batched_lstm_outs, _ = self.lstm(batched_all_ht_cpt_rel_paths_embeds)  # path_num * path_len * lstm_dim
        else:
            batched_lstm_outs = self.lstm_trans(batched_all_ht_cpt_rel_paths_embeds)  # path_num * path_len * lstm_dim
        b_idx = torch.arange(batched_lstm_outs.size(0)).to(batched_lstm_outs.device)
        batched_lstm_outs = batched_lstm_outs[b_idx, torch.cat(path_len_batched, 0) - 1, :]  # path_num * lstm_dim
        final_vecs = []
        qa_pair_cur_start = 0
        path_cur_start = 0
        # [tensor], [[h, t]], [[p],[p],[p]], [[r],[r],[r]], 3
        for e1_vec, e2_vec, ht_ids, cpt_paths, rel_paths, ht_path_num in zip(e1_vec_batched, e2_vec_batched, ht_pairs_batched,
                                                                             cpt_paths_batched, rel_paths_batched, ht_path_num_batched):
            n_qa_pairs = ht_ids.size(0)  # 1 or 0
            qa_pair_cur_end = qa_pair_cur_start + n_qa_pairs
            if n_qa_pairs == 0:
                assert len(ht_ids) == len(cpt_paths) == len(rel_paths) == 0
                # raw_qas_vecs = torch.cat([e1_vec, e2_vec, e1_vec.new_zeros(self.graph_hidden_dim*2)], 0)
                # ht_vec = self.hts_encoder(raw_qas_vecs)
                # latent_rel_vec = torch.cat((ht_vec, e1_vec.new_zeros(self.lstm_dim)), dim=0)
                latent_rel_vec = e1_vec.new_zeros(2 * self.lstm_dim)
            else:
                ht_vec = hts_vecs_batched[qa_pair_cur_start:qa_pair_cur_end].squeeze(0)  # 0, 1
                path_cur_end = path_cur_start + ht_path_num
                blo = batched_lstm_outs[path_cur_start:path_cur_end]  # 0, 3  path_num * dim
                if self.path_attention:
                    # query_vec = query_vecs_batched[qa_pair_cur_start:qa_pair_cur_end]  # 0, 1
                    query_vec = ht_vec
                    att_scores = torch.mv(blo, query_vec.squeeze(0))  # path-level attention scores
                    norm_att_scores = F.softmax(att_scores, 0)
                    latent_rel_vec_ = torch.mv(blo.t(), norm_att_scores)
                else:
                    latent_rel_vec_ = blo.mean(0)
                path_cur_start = path_cur_end  # 3
                latent_rel_vec = torch.cat((ht_vec, latent_rel_vec_), dim=-1)
            final_vecs.append(latent_rel_vec)
            qa_pair_cur_start = qa_pair_cur_end  # 1

        batched_path_out = torch.stack(final_vecs)
        batched_path_out = self.trans_fc(batched_path_out)
        if self.use_graph and training:
            return batched_path_out, losses
        else:
            return batched_path_out, None

