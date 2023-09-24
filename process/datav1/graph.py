#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2022/3/7 20:10
# @Author   : Dling
# @FileName : graph.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import torch
import networkx as nx
import itertools
import json
from tqdm import tqdm
from conceptnet import merged_relations
import numpy as np
import pickle
from multiprocessing import Pool

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


##################### neighbor graph #####################

def neighbor_relation_graph_generation(event_cpt, neighbor_list, rel_list):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple

    graph = nx.MultiDiGraph()
    attrs = set()
    for n, r in zip(neighbor_list, rel_list):
        if (event_cpt, n, r) not in attrs:
            graph.add_edge(event_cpt, n, rel=r)
            attrs.add((event_cpt, n, r))
            if r >= len(relation2id):
                graph.add_edge(n, event_cpt, rel=r - len(relation2id))
                attrs.add((n, event_cpt, r - len(relation2id)))
            else:
                graph.add_edge(n, event_cpt, rel=r + len(relation2id))
                attrs.add((n, event_cpt, r + len(relation2id)))

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
    return nx.node_link_data(g)


def generate_neighbor_graph(pruned_neighbors_path, cpnet_vocab_path, cpnet_graph_path, output_path):
    print(f'generating schema graphs for {pruned_neighbors_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet, cpnet_simple
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    nrow = sum(1 for _ in open(pruned_neighbors_path, 'r'))
    with open(pruned_neighbors_path, 'r', encoding='utf-8') as fin_pf, \
            open(output_path, 'w', encoding='utf-8') as fout:
        for line_pf in tqdm(fin_pf, total=nrow):
            doc_events = json.loads(line_pf)
            doc_graph = {'docId': doc_events['docId'], 'event_pair': []}
            for sen_events in doc_events['event_pair']:
                new_sen = []
                for e_pair in sen_events:
                    if e_pair['e1_nbs'] is not None and e_pair['e2_nbs'] is not None:
                        e1 = concept2id[e_pair['e1']['concept']]
                        e1_neigh = e_pair['e1_nbs']['neighbors']
                        e1_rel = e_pair['e1_nbs']['rel']
                        e2 = concept2id[e_pair['e2']['concept']]
                        e2_neigh = e_pair['e2_nbs']['neighbors']
                        e2_rel = e_pair['e2_nbs']['rel']
                        e1_gobj = neighbor_relation_graph_generation(e1, e1_neigh, e1_rel)
                        e2_gobj = neighbor_relation_graph_generation(e2, e2_neigh, e2_rel)

                        e1_p_neigh = e_pair['e1_nbs']['p_neighbors']['neighbors']
                        e1_p_rel = e_pair['e1_nbs']['p_neighbors']['rel']
                        e2_p_neigh = e_pair['e2_nbs']['p_neighbors']['neighbors']
                        e2_p_rel = e_pair['e2_nbs']['p_neighbors']['rel']
                        e1_p_gobj = neighbor_relation_graph_generation(e1, e1_p_neigh, e1_p_rel)
                        e2_p_gobj = neighbor_relation_graph_generation(e2, e2_p_neigh, e2_p_rel)

                        e1_n_nbs_graph, e2_n_nbs_graph = [], []
                        for n_nbs in e_pair['e1_nbs']['n_neighbors']:
                            e1_n_nbs_graph.append(
                                neighbor_relation_graph_generation(n_nbs['nb_id'], n_nbs['neighbors'], n_nbs['rel']))
                        for n_nbs in e_pair['e2_nbs']['n_neighbors']:
                            e2_n_nbs_graph.append(
                                neighbor_relation_graph_generation(n_nbs['nb_id'], n_nbs['neighbors'], n_nbs['rel']))
                    else:
                        e1_gobj, e2_gobj = None, None
                        e1_p_gobj, e2_p_gobj = None, None
                        e1_n_nbs_graph, e2_n_nbs_graph = None, None
                    pair_graph = {'e1': e_pair['e1'], 'e2': e_pair['e2'], 'e1_nbs_graph': e1_gobj, 'e1_p_nbs_graph': e1_p_gobj,
                                  'e1_n_nbs_graph': e1_n_nbs_graph, 'e2_nbs_graph': e2_gobj, 'e2_p_nbs_graph': e2_p_gobj,
                                  'e2_n_nbs_graph': e2_n_nbs_graph}
                    new_sen.append(pair_graph)
                doc_graph['event_pair'].append(new_sen)
            fout.write(json.dumps(doc_graph) + '\n')

    print(f'schema graphs saved to {output_path}')
    print()


##################### path graph #####################

def path_relation_graph_generation(paths_list, rel_list):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple

    graph = nx.MultiDiGraph()
    attrs = set()
    for p, rel in zip(paths_list, rel_list):
        assert len(rel) == len(p) - 1
        for index in range(len(p) - 1):
            h = p[index]
            t = p[index + 1]
            r = rel[index]
            if (h, t, r) not in attrs:
                graph.add_edge(h, t, rel=r)
                attrs.add((h, t, r))
                if r >= len(relation2id):
                    graph.add_edge(t, h, rel=r - len(relation2id))
                    attrs.add((t, h, r - len(relation2id)))
                else:
                    graph.add_edge(t, h, rel=r + len(relation2id))
                    attrs.add((t, h, r + len(relation2id)))

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
    return nx.node_link_data(g)


def generate_path_graph(pruned_paths_path, cpnet_vocab_path, cpnet_graph_path, output_path):
    # {"sent": s, "ans": a, "qc": question_concepts, "ac": answer_concepts}
    # {"ac": ac, "qc": qc, "pf_res": pf_res}
    # graph = nx.MultiDiGraph()
    print(f'generating schema graphs for {pruned_paths_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet, cpnet_simple
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    nrow = sum(1 for _ in open(pruned_paths_path, 'r'))
    with open(pruned_paths_path, 'r', encoding='utf-8') as fin_pf, \
            open(output_path, 'w', encoding='utf-8') as fout:
        for line_pf in tqdm(fin_pf, total=nrow):
            doc_events = json.loads(line_pf)
            doc_graph = {'docId': doc_events['docId'], 'event_pair': []}
            for sen_events in doc_events['event_pair']:
                new_sen = []
                for e_pair in sen_events:
                    event_paths = []
                    event_rel_list = []
                    if e_pair['paths'] is None:
                        gobj = None
                    else:
                        cur_paths = [item["path"] for item in e_pair['paths']]  # [[1,2,3], [1,3]]
                        cur_rels = [item["rel"] for item in e_pair['paths']]  # [[], [], []]
                        event_paths.extend(cur_paths)
                        event_rel_list.extend(cur_rels)
                        gobj = path_relation_graph_generation(event_paths, event_rel_list)
                    pair_graph = {'e1': e_pair['e1'], 'e2': e_pair['e2'], 'graph': gobj}
                    new_sen.append(pair_graph)
                doc_graph['event_pair'].append(new_sen)
            fout.write(json.dumps(doc_graph) + '\n')

    print(f'schema graphs saved to {output_path}')
    print()


if __name__ == "__main__":
    # generate_neighbor_graph('../../data/story/neighbors.pruned1.limited_6.jsonl', '../../data/cpnet/concept.txt',
    #                '../../data/cpnet/cpnet.en.pruned.graph', '../../data/story/neighbors.graph1.limited_6.jsonl')
    generate_path_graph('../../data/story/paths.pruned.limited_6.jsonl', '../../data/cpnet/concept.txt',
                        '../../data/cpnet/cpnet.en.pruned.graph', '../../data/story/paths.graph.limited_6.jsonl')
