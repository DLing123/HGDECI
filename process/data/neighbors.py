#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2022/3/6 14:27
# @Author   : Dling
# @FileName : neighbors.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
import json
import random
from scipy import spatial
import os
from conceptnet import merged_relations
import pickle

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_simple = None

concept_embs = None
relation_embs = None


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


##################### neighbors finding #####################

def get_edge(src_concept, tgt_concept):
    global cpnet
    rel_list = cpnet[src_concept][tgt_concept]  # list of dicts
    seen = set()
    res = [r['rel'] for r in rel_list.values() if
           r['rel'] not in seen and (seen.add(r['rel']) or True)]  # get unique values from rel_list
    return res


def find_neighbor_node(s_id, ifprint=False, flag='s'):
    try:
        s_neighbor = list(set(cpnet_simple[s_id]))
    except:
        print('find neighbor error')
    rl = []

    # neigh1 = ['take_rest', 'solve', 'hurt_bone', 'destroy', 'go_on_vacation', 'smash_something']
    # neigh2 = ['mend', 'tool', 'destroy', 'rectify', 'condition', 'patch']
    # new_neighbor = []

    for sn in s_neighbor:

        # temp = neigh1 if flag == 's' else neigh2
        # if id2concept[sn] not in temp:
        #     print(id2concept[sn])
        #     continue
        # new_neighbor.append(sn)

        rel_list = get_edge(s_id, sn)
        rl.append(rel_list)
        if ifprint:
            rel_list_str = []
            for rel in rel_list:
                if rel < len(id2relation):
                    rel_list_str.append(id2relation[rel])
                else:
                    rel_list_str.append(id2relation[rel - len(id2relation)] + "*")
            print(id2concept[s_id], "----[%s]---> " % ("/".join(rel_list_str)), end="")
            print(id2concept[sn], end="")
        if ifprint:
            print()
    nf_res = {"neighbors": s_neighbor, "rel": rl}
    # nf_res = {"neighbors": new_neighbor, "rel": rl}
    return nf_res


def find_neighbors_event_concept_pair(source: str, target: str, ifprint=False):
    """
    find paths for a (question concept, answer concept) pair
    source and target is text
    """
    global cpnet, cpnet_simple, concept2id, id2concept, relation2id, id2relation

    s = concept2id[source]
    t = concept2id[target]

    if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
        print(s, id2concept[s], t, id2concept[t])
        return None, None
    snf_res = find_neighbor_node(s, ifprint, 's')
    tnf_res = find_neighbor_node(t, ifprint, 't')
    return snf_res, tnf_res


def find_neighbors_event_pairs(doc):
    new_doc = {}
    nfr_ep = []
    for sen in doc[1]:
        nfr = []
        for p in sen['event_pair']:
            if p[0]['concept'] is not None and p[1]['concept'] is not None:
                nf_res = find_neighbors_event_concept_pair(p[0]['concept'], p[1]['concept'])
                nfr.append({'e1': p[0], 'e2': p[1], 'e1_nbs': nf_res[0], 'e2_nbs': nf_res[1]})  # [{}, {}, {}] a sen
            else:
                nfr.append({'e1': p[0], 'e2': p[1], 'e1_nbs': None, 'e2_nbs': None})
        nfr_ep.append(nfr)  # [[{}, {}, {} doc], [{}, {}, {} doc]]
    new_doc['docId'] = doc[0]
    new_doc['event_pair'] = nfr_ep
    return new_doc


def find_neighbors(grounded_path, cpnet_vocab_path, cpnet_graph_path, output_path, num_processes=1, random_state=0):
    print(f'generating neighbors for {grounded_path}...')
    random.seed(random_state)
    np.random.seed(random_state)

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    with open(grounded_path, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]
    data = [(item['docId'], item['sentences']) for item in data]  # [[sens], [sens], [sens], ..., [sens]]

    with Pool(num_processes) as p, open(output_path, 'w', encoding='utf-8') as fout:
        for nfr_doc in tqdm(p.imap(find_neighbors_event_pairs, data), total=len(data)):
            fout.write(json.dumps(nfr_doc) + '\n')

    print(f'paths saved to {output_path}')
    print()


##################### neighbors scoring #####################


def score_triple(h, t, r, flag):
    res = []
    for i in range(len(r)):
        if flag[i]:
            temp_h, temp_t = t, h
        else:
            temp_h, temp_t = h, t
        # result  = (cosine_sim + 1) / 2, spatial.distance.cosine是距离
        res.append((1 + 1 - spatial.distance.cosine(r[i], temp_t - temp_h)) / 2)
    return res


def score_triples(concepts, relation_id, debug=False):
    global relation_embs, concept_embs, id2relation, id2concept, concept2id
    concept_id = [concept2id[concepts[0]], concepts[1]]
    concept = concept_embs[concept_id]
    relation = []
    flag = []

    for j in range(len(relation_id)):
        if relation_id[j] >= 17:
            relation.append(relation_embs[relation_id[j] - 17])
            flag.append(1)
        else:
            relation.append(relation_embs[relation_id[j]])
            flag.append(0)
    assert concept.shape[0] == 2
    h = concept[0]
    t = concept[1]
    res = score_triple(h, t, relation, flag)

    if debug:
        print("Num of concepts:")
        print(len(concept_id))
        to_print = ""
        for i in range(concept.shape[0] - 1):
            h = id2concept[concept_id[i]]
            to_print += h + "\t"
            for rel in relation_id[i]:
                if rel >= 17:
                    # 'r-' means reverse
                    to_print += ("r-" + id2relation[rel - 17] + "/  ")
                else:
                    to_print += id2relation[rel] + "/  "
        to_print += id2concept[concept_id[-1]]
        print(to_print)
        print("Likelihood: " + str(res) + "\n")

    return res


def score_event_pairs(event_pairs):
    # {'docId':xxxxxxx, 'event_pairs': [[{'e1':{}, 'e2':{}. 'e1_nbs':{'nerbs':[], 'rel':[]}, 'e2_nbs':{} },{}], []]}
    neighbor_scores = []
    for sen in event_pairs['event_pair']:
        sen_scores = []
        for s_ep in sen:
            e1_neighbors = s_ep["e1_nbs"]
            e2_neighbors = s_ep["e2_nbs"]
            if e1_neighbors is not None and e2_neighbors is not None:
                e1_scores = []
                e2_scores = []
                assert len(e1_neighbors['neighbors']) == len(e1_neighbors['rel'])
                for n_id, rel_list in zip(e1_neighbors['neighbors'], e1_neighbors['rel']):
                    score = score_triples(concepts=(s_ep['e1']['concept'], n_id), relation_id=rel_list)
                    e1_scores.append(score)
                for n_id, rel_list in zip(e2_neighbors['neighbors'], e2_neighbors['rel']):
                    score = score_triples(concepts=(s_ep['e2']['concept'], n_id), relation_id=rel_list)
                    e2_scores.append(score)
                sen_scores.append({'e1_scores': e1_scores, 'e2_scores': e2_scores})
            else:
                sen_scores.append({'e1_scores': None, 'e2_scores': None})
        neighbor_scores.append(sen_scores)

    doc_res = {'docId': event_pairs['docId'], 'event_neighbor_score': neighbor_scores}
    return doc_res


def score_neighbors(raw_neighbors_path, concept_emb_path, rel_emb_path, cpnet_vocab_path, output_path, num_processes=1,
                    method='triple_cls'):
    print(f'scoring paths for {raw_neighbors_path}...')
    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global concept_embs, relation_embs
    if concept_embs is None:
        concept_embs = np.load(concept_emb_path)
    if relation_embs is None:
        relation_embs = np.load(rel_emb_path)

    if method != 'triple_cls':
        raise NotImplementedError()

    with open(raw_neighbors_path, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]

    with Pool(num_processes) as p, open(output_path, 'w', encoding='utf-8') as fout:
        for statement_scores in tqdm(p.imap(score_event_pairs, data), total=len(data)):
            fout.write(json.dumps(statement_scores) + '\n')

    print(f'path scores saved to {output_path}')
    print()


##################### neighbors pruning #####################
def sort_neighbors(event_info, score_info, tag, threshold):
    def limited_knows(rel2nid, num):  # [rel: nid]
        cp_ids, rel_ids, scores = [], [], []
        if num == 1:
            for r in rel2nid:
                for cid in rel2nid[r]:
                    cp_ids.append(cid[0])
                    rel_ids.append(r)
                    scores.append(cid[1])
        else:
            # causes isa capableof usedfor hassubevent createdby desires relatedto partof madeof notcapableof notdesires
            # hasproperty receivesaction atlocation antonym hascontext
            rel_list = [3, 20, 5, 22, 2, 19, 16, 33, 7, 24, 4, 21, 6, 23, 15, 32, 8, 25, 11, 28, 12, 29, 13, 30, 10, 27,
                        14, 31, 1, 18, 0, 17, 9, 26]
            for rel in rel_list:
                if len(cp_ids) >= 6:
                    break
                else:
                    if rel in rel2nid:
                        for cid in rel2nid[rel]:
                            if len(cp_ids) >= 6:
                                break
                            else:
                                if cid[0] not in cp_ids and cid[1] > 0.45:
                                    cp_ids.append(cid[0])
                                    rel_ids.append(rel)
                                    scores.append(cid[1])
            assert len(cp_ids) == len(rel_ids) == len(scores)
        return cp_ids, rel_ids, scores

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources('/home/dl/data/cpnet/concept.txt')
    if event_info[tag + '_nbs'] is not None:
        rel2nid = {}
        for n_id, r_list, s_list in zip(event_info[tag + '_nbs']['neighbors'], event_info[tag + '_nbs']['rel'],
                                        score_info[tag + '_scores']):
            for i, r in enumerate(r_list):
                if r not in rel2nid:
                    rel2nid[r] = []
                if (n_id, s_list[i]) not in rel2nid[r]:
                    rel2nid[r].append((n_id, s_list[i]))

        num = 0
        for r in rel2nid:
            if r < 17:
                thres_temp = threshold + 1
            else:
                if len(rel2nid) == 1:
                    thres_temp = threshold + 1
                else:
                    thres_temp = threshold
            rel2nid[r] = sorted(rel2nid[r], key=lambda l: l[1], reverse=True)[:thres_temp]
            num += len(rel2nid[r])

        neighbors, rel, scores = limited_knows(rel2nid, num)
        return {'neighbors': neighbors, 'rel': rel, 'scores': scores}
    else:
        return None


def prune_neighbors(raw_neigh_path, neigh_scores_path, output_path, threshold, verbose=True):
    print(f'pruning neighbors for {raw_neigh_path}...')
    ori_num = 0
    pruned_num = 0
    nrow = sum(1 for _ in open(raw_neigh_path, 'r'))
    with open(raw_neigh_path, 'r', encoding='utf-8') as fin_raw, \
            open(neigh_scores_path, 'r', encoding='utf-8') as fin_score, \
            open(output_path, 'w', encoding='utf-8') as fout:
        for line_raw, line_score in tqdm(zip(fin_raw, fin_score), total=nrow):
            event_pairs = json.loads(line_raw)
            event_scores = json.loads(line_score)
            for s_pairs, s_scores in zip(event_pairs['event_pair'], event_scores['event_neighbor_score']):
                for e_item, s_item in zip(s_pairs, s_scores):  # 每句话中的每个事件对
                    e1_nbs = sort_neighbors(e_item, s_item, 'e1', threshold)
                    e2_nbs = sort_neighbors(e_item, s_item, 'e2', threshold)
                    if e1_nbs is None:
                        assert e2_nbs is None
                    if e1_nbs is not None and e2_nbs is not None:
                        ori_num += len(e_item['e1_nbs']['neighbors']) + len(e_item['e2_nbs']['neighbors'])
                        pruned_num += len(e1_nbs['neighbors']) + len(e2_nbs['neighbors'])
                    e_item['e1_nbs'] = e1_nbs
                    e_item['e2_nbs'] = e2_nbs
            fout.write(json.dumps(event_pairs) + '\n')

    if verbose:
        print("ori_len: {}   pruned_len: {}   keep_rate: {:.4f}".format(ori_num, pruned_num, pruned_num / ori_num))

    print(f'pruned paths saved to {output_path}')
    print()


if __name__ == "__main__":
    # find_neighbors('/home/dl/data/story/ground_res_c.jsonl', '/home/dl/data/cpnet/concept.txt',
    #                '/home/dl/data/cpnet/cpnet.en.pruned.graph',
    #                '/home/dl/data/story/neighbors.raw.jsonl')
    # score_neighbors('/home/dl/data/story/neighbors.raw.jsonl', '/home/dl/data/transe/glove.transe.sgd.ent.npy',
    #                 '/home/dl/data/transe/glove.transe.sgd.rel.npy', '/home/dl/data/cpnet/concept.txt',
    #                 '/home/dl/data/story/neighbors.scores.jsonl')
    # prune_neighbors('/home/dl/data/story/neighbors.raw.jsonl', '/home/dl/data/story/neighbors.scores.jsonl',
    #                 '/home/dl/data/story/neighbors.pruned1.limited_6.jsonl', 1)
    load_resources('/home/dl/data/cpnet/concept.txt')
    load_cpnet('/home/dl/data/cpnet/cpnet.en.pruned.graph')
    find_neighbor_node(concept2id['break'], ifprint=True)
