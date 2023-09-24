#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2022/3/10 15:16
# @Author   : Dling
# @FileName : dataloader.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import dgl.function as fn

import json
import os
import torch
import pickle
import dgl
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch.utils.data as data
from sklearn.model_selection import KFold
from transformers import BertTokenizer
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths
import random


def negative_sampling(data, ratio=0.6):
    result = []
    for d in data:
        if d[-1] == 0:
            if random.random() < ratio:
                continue
        result.append(d)
    return result


def split_story_line(infile, use_topic=False):
    def _split(total_num):
        X = np.arange(total_num)
        split_index = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            split_index.append((train_index, test_index))
        return split_index

    with open(infile, 'r', encoding='utf-8') as fin_raw:
        examples = [json.loads(line) for line in fin_raw]
        examples = np.array(examples, dtype=object)
        if use_topic:
            train_dataset, test_dataset = [], []
            all_topic = np.array(['1', '3', '4', '5', '7', '8', '12', '13', '14', '16', '18', '19', '20',
                                  '22', '23', '24', '30', '32', '33', '35', '37', '41'])
            split_index = _split(len(all_topic))
            train_index, test_index = split_index[4]
            # train_index, test_index = np.array([10]), np.array([-4])
            for ex in examples:
                topic = ex[0].split('/')[0].split('_')[0]
                if topic in all_topic[train_index]:
                    train_dataset.append(ex)
                elif topic in all_topic[test_index]:
                    test_dataset.append(ex)
                else:
                    print('topic error', topic)
            # train_dataset = negative_sampling(train_dataset, 0.8)
        else:
            split_index = _split(len(examples))
            train_index, test_index = split_index[0]
            train_dataset = examples[train_index]
            test_dataset = examples[test_index]
            # train_dataset = negative_sampling(train_dataset, 0.8)
        fin_raw.close()
    return train_dataset, test_dataset


class MultiGPUNxgDataBatchGenerator(object):
    def __init__(self, device0, device1, batch_size, example_ids, labels,
                 text_data, path_list, neigh_graph=None, path_graph=None, epoch_subs=None, training=True):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.example_ids = example_ids
        self.indexes = torch.randperm(len(self.example_ids))
        self.labels = labels
        self.text_data = text_data
        self.path_list = path_list
        self.neigh_graph = neigh_graph
        self.path_graph = path_graph
        self.epoch_subs = epoch_subs
        self.epoch_num = 0
        self.training = training

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        subg_maps = {} if self.epoch_subs else None
        if self.training:
            self.epoch_num += 1
            print('len_subg_maps', len(subg_maps), flush=True)
            print(self.epoch_num, flush=True)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_ids = [self.example_ids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device0)
            batch_text_data = [self._to_device(x[batch_indexes], self.device0) for x in self.text_data]

            batch_path_lists = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.path_list]
            path_graph_data = [(self.path_graph[0][i], self.path_graph[1][i]) for i in batch_indexes]
            batched_path_graph, concept_mapping_dicts, rel_mapping_dicts, batched_sub1_graph,\
            mask_sg1s, subg_maps = self._trans_path_graph(path_graph_data, self.epoch_num, subg_maps)

            neigh_graph_data = [self.neigh_graph[i] for i in batch_indexes]  # [[dg1, dg2], [], [dg1, dg2], []]
            batched_neigh_data, batched_neigh_flag,  batched_nsub_graph, batched_nmask_sgs, subg_maps = \
                self._trans_neigh_graph(neigh_graph_data, subg_maps)
            yield tuple([batch_ids, batch_labels, batch_text_data, batch_path_lists, batched_path_graph,
                         concept_mapping_dicts, rel_mapping_dicts, batched_sub1_graph, mask_sg1s,
                         batched_neigh_data, batched_neigh_flag, batched_nsub_graph, batched_nmask_sgs])
        if self.training:
            print('epoch finish len_subg_maps', len(subg_maps), flush=True)

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)

    def _create_pathsub_graph(self, ori_g, s_len, flag=1, subg_maps=None):
        def _created(s_len, flag):
            num_nodes = len(ori_g.nodes())
            if flag == 1:
                index_list = list(range(0, s_len))
            else:
                node_list = list(range(0, s_len))[1: s_len-1] + list(range(s_len, num_nodes))
                index_list = sorted([0, s_len-1] + random.sample(node_list, max(int(len(node_list) * 0.73), 0)))
            index = torch.tensor(index_list)
            sg = ori_g.subgraph(index)
            mask_sg = torch.zeros(num_nodes)
            mask_sg[index] = 1
            return sg, mask_sg
        # def _created(s_len, flag):
        #     ori_g.out_degrees()
        #     num_nodes = len(ori_g.nodes())
        #     if flag == 1:
        #         index_list = list(range(0, s_len))
        #         index = torch.tensor(index_list)
        #         sg = ori_g.subgraph(index)
        #         mask_sg = torch.zeros(num_nodes).to(torch.bool)
        #         mask_sg[index] = True
        #     else:
        #         in_degrees = ori_g.in_degrees()
        #         out_degrees = ori_g.out_degrees()
        #         assert in_degrees.equal(out_degrees)
        #         deg_sco = torch.log(out_degrees)
        #         deg_weights = (deg_sco - deg_sco.min()) / (deg_sco.max() - deg_sco.min())
        #         deg_weights = deg_weights.where(deg_weights > 0.65, torch.ones_like(deg_weights) * 0.65)  # p_tau
        #         weights = (1 - 0.2) / torch.mean(deg_weights) * deg_weights
        #         weights = weights.where(weights < 0.95, torch.ones_like(weights) * 0.95)
        #         weights[0], weights[s_len - 1] = 1, 1
        #         # print(torch.mean(weights))
        #         # print(torch.min(weights))
        #         # print(torch.max(weights))
        #         mask_sg = torch.bernoulli(weights).to(torch.bool)
        #         index = mask_sg.nonzero().view(-1)
        #         sg = ori_g.subgraph(index)
        #     return sg, mask_sg
        ep_id = (ori_g.ndata["cncpt_ids"][0].item(), ori_g.ndata["cncpt_ids"][s_len - 1].item(), flag)
        if subg_maps is not None:
            if ep_id in subg_maps:
                sg, mask_sg = subg_maps[ep_id]
            else:
                sg, mask_sg = _created(s_len, flag)
                subg_maps[ep_id] = (sg, mask_sg)
        else:
            sg, mask_sg = _created(s_len, flag)
        return sg, mask_sg, subg_maps

    def _trans_path_graph(self, graph_data, epoch_num=0, subg_maps=None):
        flat_graph_data = []
        concept_mapping_dicts, rel_mapping_dicts = [], []
        acc_start_p, acc_start_r = 0, 0
        flat_sg1s, mask_sg1s = [], []
        flat_sg2s, mask_sg2s = [], []
        for g, s_len in graph_data:
            concept_mapping_dict = {}
            rel_mapping_dict = {}
            if isinstance(g, list):
                assert len(g) == s_len == 0
            else:
                for index, cncpt_id in enumerate(g.ndata['cncpt_ids']):
                    concept_mapping_dict[int(cncpt_id)] = acc_start_p + index
                acc_start_p += len(g.nodes())
                e_u, e_v = g.edges()
                for index, edge in enumerate(zip(e_u, e_v)):
                    u, v = edge
                    rel_k = (int(g.ndata['cncpt_ids'][u.item()]), int(g.ndata['cncpt_ids'][v.item()]))
                    rel_mapping_dict[rel_k] = acc_start_r + index
                acc_start_r += len(e_u)
                flat_graph_data.append(g)
                if epoch_num < 3:
                    sg1, mask_sg1, subg_maps = self._create_pathsub_graph(g, s_len, 1, subg_maps)
                else:
                    sg1, mask_sg1, subg_maps = self._create_pathsub_graph(g, s_len, 2, subg_maps)
                flat_sg1s.append(sg1)  # , flat_sg2s.append(sg2)
                mask_sg1s.append(mask_sg1.to(self.device0))
                # mask_sg2s.append(mask_sg2.to(self.device0))
            concept_mapping_dicts.append(concept_mapping_dict)
            rel_mapping_dicts.append(rel_mapping_dict)
        batched_graph = dgl.batch(flat_graph_data).to(self.device0)
        batched_sub1_graph = dgl.batch(flat_sg1s).to(self.device0)
        # batched_sub2_graph = dgl.batch(flat_sg2s).to(self.device0)
        return batched_graph, concept_mapping_dicts, rel_mapping_dicts, batched_sub1_graph, \
               mask_sg1s, subg_maps

    def _create_neisub_graph(self, ori_g, subg_maps=None):
        def _created():
            num_nodes = len(ori_g.nodes())
            index_list = sorted([0] + random.sample(list(range(1, num_nodes)), max(int(num_nodes * 0.6), 0)))
            index = torch.tensor(index_list)
            sg = ori_g.subgraph(index)
            mask_sg = torch.zeros(num_nodes)
            mask_sg[index] = 1
            return sg, mask_sg
        # def _created():
        #     nei_num = int(len(ori_g.edata["edge_sco"])/2)
        #     exp_sco = torch.exp(ori_g.edata["edge_sco"][:nei_num])
        #     sco_weights = (exp_sco - exp_sco.min()) / (exp_sco.max() - exp_sco.min())
        #     sco_weights = sco_weights.where(sco_weights > 0.65, torch.ones_like(sco_weights) * 0.65)
        #     weights = (1-0.2) / torch.mean(sco_weights) * sco_weights
        #     weights = weights.where(weights < 0.95, torch.ones_like(weights) * 0.95)
        #     # print(torch.mean(weights))
        #     # print(torch.min(weights))
        #     # print(torch.max(weights))
        #     mask_sg = torch.cat([torch.tensor([1.]), torch.bernoulli(weights)]).to(torch.bool)
        #     index = mask_sg.nonzero().view(-1)
        #     sg = ori_g.subgraph(index)
        #     return sg, mask_sg
        if subg_maps is not None:
            e_id = ori_g.ndata["cncpt_ids"][0].item()
            if e_id in subg_maps:
                sg, mask_sg = subg_maps[e_id]
            else:
                sg, mask_sg = _created()
                subg_maps[e_id] = (sg, mask_sg)
        else:
            sg, mask_sg = _created()
        return sg, mask_sg, subg_maps

    def _trans_neigh_graph(self, graph_data, subg_maps):
        graph_flag, flat_graph_data = [], []
        flat_sgs, mask_sgs = [], []
        for n_pair in graph_data:
            if isinstance(n_pair, list) and len(n_pair) == 0:
                graph_flag.append([0, 0])
            else:
                flat_graph_data.append(n_pair[0])
                flat_graph_data.append(n_pair[1])
                graph_flag.append([1, 1])
                sg1, mask_sg1, subg_maps = self._create_neisub_graph(n_pair[0], subg_maps)
                sg2, mask_sg2, subg_maps = self._create_neisub_graph(n_pair[1], subg_maps)
                flat_sgs.append(sg1)
                flat_sgs.append(sg2)
                mask_sgs.append(mask_sg1.to(self.device1))
                mask_sgs.append(mask_sg2.to(self.device1))
        batched_graph = dgl.batch(flat_graph_data).to(self.device1)
        batched_sub_graph = dgl.batch(flat_sgs).to(self.device1)
        return batched_graph, graph_flag, batched_sub_graph, mask_sgs, subg_maps


class EciDataLoader(data.Dataset):
    def __init__(self, config):
        super(EciDataLoader, self).__init__()
        self.batch_size = config['batch_size']
        self.max_seq_len = config['max_seq_len']
        self.max_path_len = config['max_path_len']
        self.pretrained_model_path = config['pretrained_model_path']
        self.device0 = config['device0']
        self.device1 = config['device1']
        self.epoch_subs = config['epoch_subs']

    def _load_input_tensors(self, dataset):
        class InputExample(object):
            def __init__(self, example_id, text, e1_pos, e2_pos, label=None):
                self.example_id = example_id
                self.text = text
                self.e1_pos = [int(i) for i in e1_pos.split('_')]
                self.e2_pos = [int(i) for i in e2_pos.split('_')]
                self.label = label

        def convert_examples_to_features(examples, tokenizer):
            example_ids, sentences, events1, events2, labels = list(), list(), list(), list(), list()
            for ex in examples:
                ex.text = ['[CLS]'] + ex.text[:ex.e1_pos[0]] + ['[unused0]'] + ex.text[ex.e1_pos[0]:ex.e1_pos[-1]+1] + \
                          ['[unused1]'] + ex.text[ex.e1_pos[-1]+1:ex.e2_pos[0]] + ['[unused2]'] + \
                          ex.text[ex.e2_pos[0]:ex.e2_pos[-1]+1] + ['[unused3]'] + ex.text[ex.e2_pos[-1]+1:] + ['[SEP]']
                ex.e1_pos = list(map(lambda x: x + 2, ex.e1_pos))
                ex.e1_pos = [ex.e1_pos[0] - 1] + ex.e1_pos + [ex.e1_pos[-1] + 1]
                ex.e2_pos = list(map(lambda x: x + 4, ex.e2_pos))
                ex.e2_pos = [ex.e2_pos[0] - 1] + ex.e2_pos + [ex.e2_pos[-1] + 1]
                sentence_vec = []
                e1_vec = []
                e2_vec = []
                for i, w in enumerate(ex.text):
                    tokens = tokenizer.tokenize(w) if w not in (
                        "[CLS]", "[unused0]", "[unused1]", "[unused2]", "[unused3]", "[SEP]") else [w]
                    xx = tokenizer.convert_tokens_to_ids(tokens)
                    if i in ex.e1_pos:
                        e1_vec.extend(list(range(len(sentence_vec), len(sentence_vec) + len(xx))))
                    if i in ex.e2_pos:
                        e2_vec.extend(list(range(len(sentence_vec), len(sentence_vec) + len(xx))))
                    sentence_vec.extend(xx)
                example_ids.append(ex.example_id)
                sentences.append(sentence_vec)
                events1.append(e1_vec)
                events2.append(e2_vec)
                labels.append(ex.label)

            sentence_len = [len(s_vec) for s_vec in sentences]
            max_sentence_len = min(max(sentence_len), self.max_seq_len)
            event1_lens = [len(e1_vec) for e1_vec in events1]
            event2_lens = [len(e2_vec) for e2_vec in events2]
            max_event_len = min(max(event1_lens + event2_lens), max_sentence_len)

            sentences = list(map(lambda x: pad_sequence_to_length(x, max_sentence_len), sentences))
            events1 = list(map(lambda x: pad_sequence_to_length(x, max_event_len), events1))
            events2 = list(map(lambda x: pad_sequence_to_length(x, max_event_len), events2))
            sentences = torch.LongTensor(sentences)
            events1 = torch.LongTensor(events1)
            events2 = torch.LongTensor(events2)
            labels = torch.LongTensor(labels)

            mask_sentences = get_mask_from_sequence_lengths(torch.LongTensor(sentence_len), max_sentence_len)
            mask_events1 = get_mask_from_sequence_lengths(torch.LongTensor(event1_lens), max_event_len)
            mask_events2 = get_mask_from_sequence_lengths(torch.LongTensor(event2_lens), max_event_len)

            return example_ids, labels, sentences, mask_sentences, events1, mask_events1, events2, mask_events2

        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        examples = []
        for ex in dataset:
            ex_id, text, e1_pos, e2_pos, label = ex[0], ex[1], ex[2], ex[3], ex[-1]
            examples.append(InputExample(example_id=ex_id, text=text, e1_pos=e1_pos, e2_pos=e2_pos, label=label))
        example_ids, labels, *data_tensor = convert_examples_to_features(examples, tokenizer)
        return (example_ids, labels, *data_tensor)

    def _load_paths_data(self, dataset):
        # [example_id, text, e1_pos, e2_pos, e_neigh_g, e_path_p, e_path_g, label]
        ht_pair_data, cpt_path_data, rel_path_data, path_len_data, path_num_data = [], [], [], [], []
        for ex in dataset:
            ht_pair, paths, rels, path_len = [], [], [], []
            ep_path = ex[5]
            if ep_path['paths'] is not None and len(ep_path['paths']) > 0:
                temp = ep_path['paths'][0]['path']
                ht_pair.append([temp[0], temp[-1]])
                path_num = len(ep_path['paths'])
                for item in ep_path['paths']:
                    p = item['path']
                    r = item['rel']
                    if len(p) > self.max_path_len:
                        continue
                    assert len(p) - 1 == len(r)
                    path_len.append(len(p))
                    p += [0] * (self.max_path_len - len(p))  # padding
                    r += [0] * (self.max_path_len - len(r))  # padding
                    paths.append(p)
                    rels.append(r)
            else:
                path_num = 0
            cpt_path_data.append(torch.tensor(paths) if paths else torch.zeros((0, self.max_path_len), dtype=torch.int64))
            rel_path_data.append(torch.tensor(rels) if rels else torch.zeros((0, self.max_path_len), dtype=torch.int64))
            path_len_data.append(torch.tensor(path_len) if path_len else torch.zeros(0, dtype=torch.int64))
            ht_pair_data.append(torch.tensor(ht_pair) if ht_pair else torch.zeros((0, 2), dtype=torch.int64))
            path_num_data.append(torch.tensor(path_num, dtype=torch.int64))

        return ht_pair_data, cpt_path_data, rel_path_data, path_len_data, path_num_data  # [tensor, tensor]

    def _load_graph(self, nxg, n_flag=False):
        dg = dgl.from_networkx(nxg)
        cids = [nxg.nodes[n_id]['cid'] for n_id in range(len(dg.nodes()))]
        dg.ndata.update({'cncpt_ids': torch.tensor(cids)})
        rels = [nxg.edges[u, v, k]['rel'] for u, v, k in nxg.edges]
        dg.edata.update({'cncpt_rel': torch.tensor(rels)})
        if n_flag:
            scos = [nxg.edges[u, v, k]['sco'] for u, v, k in nxg.edges]
            dg.edata.update({'edge_sco': torch.tensor(scos)})
        for u, v, k in nxg.edges:
            assert abs(nxg.edges[u, v, k]['rel'] - nxg.edges[v, u, k]['rel']) == 17
        return dg

    def _load_neigh_graph(self, dataset):
        # [example_id, text, e1_pos, e2_pos, e_neigh_g, e_path_p, e_path_g, label]
        dgs = []
        for ex in dataset:
            ep_graph = ex[4]
            if ep_graph['e1_nbs_graph'] is not None and ep_graph['e2_nbs_graph'] is not None:
                assert len(ep_graph['e1_nbs_graph']['nodes']) > 0
                assert len(ep_graph['e2_nbs_graph']['nodes']) > 0
                nxg1 = nx.node_link_graph(ep_graph['e1_nbs_graph'])
                dg1 = self._load_graph(nxg1, True)
                nxg2 = nx.node_link_graph(ep_graph['e2_nbs_graph'])
                dg2 = self._load_graph(nxg2, True)
                dgs.append([dg1, dg2])
            else:
                dgs.append([])
        return dgs

    def _load_paths_graphs(self, dataset):
        # [example_id, text, e1_pos, e2_pos, e_neigh_g, e_path_p, e_path_g, label]
        dgs = []
        s_lens = []
        path_num = set()
        for ex in dataset:
            ep_graph = ex[6]
            if ep_graph['graph'] is not None and len(ep_graph['graph']['nodes']) > 0:
                nxg = nx.node_link_graph(ep_graph['graph'])
                dg = self._load_graph(nxg)
                dgs.append(dg)
                s_lens.append(ep_graph['s_len'])
                path_num.add((ep_graph['graph']['nodes'][0]["cid"], ep_graph['graph']['nodes'][ep_graph['s_len']-1]["cid"], 1))
                path_num.add((ep_graph['graph']['nodes'][0]["cid"], ep_graph['graph']['nodes'][ep_graph['s_len']-1]["cid"], 2))
            else:
                dgs.append([])
                s_lens.append(0)
        return dgs, s_lens

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, index):
        pass

    def load_batch(self, dataset, training=True):
        self.example_ids, self.labels, *self.text_data = self._load_input_tensors(dataset)
        self.neigh_graph_data = self._load_neigh_graph(dataset)
        self.path_list_data = list(self._load_paths_data(dataset))
        self.path_graph_data = self._load_paths_graphs(dataset)
        return MultiGPUNxgDataBatchGenerator(self.device0, self.device1, self.batch_size, self.example_ids, self.labels,
                                    self.text_data, self.path_list_data, self.neigh_graph_data, self.path_graph_data, self.epoch_subs, training)

