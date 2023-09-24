#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2022/3/6 11:41
# @Author   : Dling
# @FileName : paths.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
import json
import random
import os
from conceptnet import merged_relations
from scipy import spatial
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


##################### paths finding #####################

def get_edge(src_concept, tgt_concept):
    global cpnet
    rel_list = cpnet[src_concept][tgt_concept]  # list of dicts
    seen = set()
    res = [r['rel'] for r in rel_list.values() if r['rel'] not in seen and (seen.add(r['rel']) or True)]  # get unique values from rel_list
    return res


def find_paths_event_concept_pair(source: str, target: str, ifprint=False):
    """
    find paths for a (question concept, answer concept) pair
    source and target is text
    """
    global cpnet, cpnet_simple, concept2id, id2concept, relation2id, id2relation

    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources('/home/dl/data/cpnet/concept.txt')
    if cpnet is None or cpnet_simple is None:
        load_cpnet('/home/dl/data/cpnet/cpnet.en.pruned.graph')

    s = concept2id[source]
    t = concept2id[target]

    if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
        print(s, id2concept[s], t, id2concept[t])
        return None

    all_path = []
    try:
        for p in nx.shortest_simple_paths(cpnet_simple, source=s, target=t):
            if len(p) <= 6:  # top 30 paths
                if len(all_path) < 30:
                    all_path.append(p)
                else:
                    break
            else:
                break
    except nx.exception.NetworkXNoPath:
        pass

    pf_res = []
    # temps = [['break', 'mend', 'repair'],
    #          ['break', 'mend', 'fix', 'repair'],
    #          ['break', 'destroy', 'fix', 'repair'],
    #          ['break', 'fill', 'order', 'repair'],
    #          ['break', 'smash', 'fix', 'repair'],
    #          ['break', 'smash', 'destroy', 'fix', 'repair'],
    #          ['break', 'point', 'repair']]
    for p in all_path:

        # temp = []
        # for item in p:
        #     temp.append(id2concept[item])
        # print(temp)
        # if temp not in temps:
        #     continue

        rl = []
        for src in range(len(p) - 1):
            src_concept = p[src]
            tgt_concept = p[src + 1]
            rel_list = get_edge(src_concept, tgt_concept)
            rl.append(rel_list)

            if ifprint:
                rel_list_str = []
                for rel in rel_list:
                    if rel < len(id2relation):
                        rel_list_str.append(id2relation[rel])
                    else:
                        rel_list_str.append(id2relation[rel - len(id2relation)] + "*")

                print(id2concept[p[src]], "----[%s]---> " % ("/".join(rel_list_str)), end="")
                print(id2concept[p[src+1]], end="")
        if ifprint:
                print()

        pf_res.append({"path": p, "rel": rl})

    return pf_res


def find_paths_event_pair(doc):
    new_doc = {}
    pfr_ep = []
    for sen in doc[1]:
        pfr = []
        for p in sen['event_pair']:
            if p[0]['concept'] is not None and p[1]['concept'] is not None:
                if p[0]['concept'] == p[1]['concept']:
                    pfr.append({'e1': p[0], 'e2': p[1], 'paths': None})
                else:
                    pf_res = find_paths_event_concept_pair(p[0]['concept'], p[1]['concept'])
                    if len(pf_res) == 0:
                        pf_res = None
                    pfr.append({'e1': p[0], 'e2': p[1], 'paths': pf_res})  # [{}, {}, {}] a sen
            else:
                pfr.append({'e1': p[0], 'e2': p[1], 'paths': None})
        pfr_ep.append(pfr)  # [[{}, {}, {} doc], [{}, {}, {} doc]]
    new_doc['docId'] = doc[0]
    new_doc['event_pair'] = pfr_ep
    return new_doc


def find_paths(grounded_path, cpnet_vocab_path, cpnet_graph_path, output_path, num_processes=1, random_state=0):
    print(f'generating paths for {grounded_path}...')
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
        for pfr_qa in tqdm(p.imap(find_paths_event_pair, data), total=len(data)):
            fout.write(json.dumps(pfr_qa) + '\n')

    print(f'paths saved to {output_path}')
    print()


##################### path scoring #####################

def score_triple(h, t, r, flag):
    res = -10
    for i in range(len(r)):
        if flag[i]:
            temp_h, temp_t = t, h
        else:
            temp_h, temp_t = h, t
        # result  = (cosine_sim + 1) / 2
        res = max(res, (1 + 1 - spatial.distance.cosine(r[i], temp_t - temp_h)) / 2)
    return res


def score_triples(concept_id, relation_id, debug=False):
    global relation_embs, concept_embs, id2relation, id2concept
    concept = concept_embs[concept_id]
    relation = []
    flag = []
    for i in range(len(relation_id)):
        embs = []
        l_flag = []

        if 0 in relation_id[i] and 17 not in relation_id[i]:
            relation_id[i].append(17)
        elif 17 in relation_id[i] and 0 not in relation_id[i]:
            relation_id[i].append(0)
        if 15 in relation_id[i] and 32 not in relation_id[i]:
            relation_id[i].append(32)
        elif 32 in relation_id[i] and 15 not in relation_id[i]:
            relation_id[i].append(15)

        for j in range(len(relation_id[i])):
            if relation_id[i][j] >= 17:
                embs.append(relation_embs[relation_id[i][j] - 17])
                l_flag.append(1)
            else:
                embs.append(relation_embs[relation_id[i][j]])
                l_flag.append(0)
        relation.append(embs)
        flag.append(l_flag)

    res = 1
    for i in range(concept.shape[0] - 1):
        h = concept[i]
        t = concept[i + 1]
        score = score_triple(h, t, relation[i], flag[i])
        res *= score

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
    # pf_res = [ {"path":[1, 2, 3], "rel":[[r1_before, r2], [r4]]},
    #            {"path":[1, 0, 3], "rel":[[r3], [r4]]} ]
    path_scores = []
    for sen in event_pairs['event_pair']:
        sen_scores = []
        for s_ep in sen:
            statement_paths = s_ep['paths']
            if statement_paths is not None:
                p_scores = []
                for path in statement_paths:
                    assert len(path["path"]) > 1
                    score = score_triples(concept_id=path["path"], relation_id=path["rel"])
                    p_scores.append(score)
                assert len(p_scores) == len(statement_paths)
                sen_scores.append(p_scores)
            else:
                sen_scores.append(None)
        path_scores.append(sen_scores)

    doc_res = {'docId': event_pairs['docId'], 'event_path_score': path_scores}
    return doc_res


def score_paths(raw_paths_path, concept_emb_path, rel_emb_path, cpnet_vocab_path, output_path, num_processes=1, method='triple_cls'):
    print(f'scoring paths for {raw_paths_path}...')
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

    with open(raw_paths_path, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]

    with Pool(num_processes) as p, open(output_path, 'w', encoding='utf-8') as fout:
        for statement_scores in tqdm(p.imap(score_event_pairs, data), total=len(data)):
            fout.write(json.dumps(statement_scores) + '\n')

    print(f'path scores saved to {output_path}')
    print()


def prune_paths(raw_paths_path, path_scores_path, output_path, thres_num, verbose=True):
    print(f'pruning paths for {raw_paths_path}...')
    ori_len = 0
    pruned_len = 0
    nrow = sum(1 for _ in open(raw_paths_path, 'r'))
    with open(raw_paths_path, 'r', encoding='utf-8') as fin_raw, \
            open(path_scores_path, 'r', encoding='utf-8') as fin_score, \
            open(output_path, 'w', encoding='utf-8') as fout:
        for line_raw, line_score in tqdm(zip(fin_raw, fin_score), total=nrow):
            event_pairs = json.loads(line_raw)
            event_pairs_scores = json.loads(line_score)
            for s_pairs, s_scores in zip(event_pairs['event_pair'], event_pairs_scores['event_path_score']):
                for e_item, s_item in zip(s_pairs, s_scores):
                    ori_paths = e_item['paths']
                    if ori_paths is not None:
                        assert len(ori_paths) == len(s_item)
                        # pruned_paths = [p for p, s in zip(ori_paths, s_item) if s >= threshold]
                        path_temp = [(p, s) for p, s in zip(ori_paths, s_item)]
                        sorted_path = sorted(path_temp, key=lambda l: l[1], reverse=True)
                        assert len(sorted_path) == len(ori_paths)
                        pruned_paths = [ori_paths[0]]
                        for item in sorted_path:
                            if len(pruned_paths) >= thres_num:
                                break
                            else:
                                if item[0] not in pruned_paths:
                                    pruned_paths.append(item[0])
                        ori_len += len(ori_paths)
                        pruned_len += len(pruned_paths)
                        assert len(ori_paths) >= len(pruned_paths)
                        e_item['paths'] = pruned_paths
                        if len(pruned_paths) < thres_num:
                            print("**less thres_num", len(pruned_paths), event_pairs['docId'])

            fout.write(json.dumps(event_pairs) + '\n')

    if verbose:
        print("ori_len: {}   pruned_len: {}   keep_rate: {:.4f}".format(ori_len, pruned_len, pruned_len / ori_len))

    print(f'pruned paths saved to {output_path}')
    print()


if __name__ == "__main__":
    # find_paths('/home/dl/data/story/ground_res_c.jsonl', '/home/dl/data/cpnet/concept.txt',
    #            '/home/dl/data/cpnet/cpnet.en.pruned.graph', '/home/dl/data/story/paths.raw.jsonl')
    # score_paths('/home/dl/data/story/paths.raw.jsonl', '/home/dl/data/transe/glove.transe.sgd.ent.npy',
    #             '/home/dl/data/transe/glove.transe.sgd.rel.npy', '/home/dl/data/cpnet/concept.txt',
    #             '/home/dl/data/story/paths.scores.jsonl')
    prune_paths('/home/dl/data/story/paths.raw.jsonl', '/home/dl/data/story/paths.scores.jsonl',
                    '/home/dl/data/story/paths.pruned.limited_4.jsonl', 4)
    # find_paths_event_concept_pair('break', 'repair', True)
