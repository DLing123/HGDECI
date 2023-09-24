#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2022/3/4 09:49
# @Author   : Dling
# @FileName : grounding.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import spacy
from tqdm import tqdm
import json
import nltk
from multiprocessing import Pool
from spacy.matcher import Matcher


CPNET_VOCAB = None


def load_concept(file_name):
    vocab = []
    with open(file_name, 'r') as fin:
        for line in fin:
            vocab.append(line.strip())
    return vocab


def find_concept(event, vocab_path):
    global CPNET_VOCAB
    if CPNET_VOCAB is None:
        CPNET_VOCAB = load_concept(vocab_path)
    if event in CPNET_VOCAB:
        return 1
    else:
        return 0


def ground(statement_path, vocab_path, output_path, debug=False):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    with open(statement_path, 'r', encoding='utf-8') as fin:
        lines = [line for line in fin]
    fin.close()

    if debug:
        lines = lines[192:193]

    docs = []
    for line in lines:
        if line == "":
            continue
        j = json.loads(line)
        for sen in j['sentences']:
            for event in sen['events']:
                tag1, tag2 = 0, 0
                e_text = event['tokens'].replace(' ', '_').replace('_-_', '_').strip('"').strip("'")
                if e_text == 'callup':
                    e_text = 'call_up'
                if e_text == 'pricecutting':
                    e_text = 'price_cutting'
                if e_text == 'profittaking':
                    e_text = 'profit_taking'
                if e_text == 'complies':
                    e_text = 'comply'
                if e_text == 'occupies':
                    e_text = 'occupy'
                event_other = None
                if len(e_text.split('_')) == 1:
                    doc = nlp(e_text)
                    event_l = doc[0].lemma_
                    tag1 = find_concept(event_l, vocab_path)
                    if not tag1:
                        tag2 = find_concept(e_text, vocab_path)
                else:
                    assert len(e_text.split('_')) > 1
                    et = nlp(e_text.replace('_', ' '))
                    temp = []
                    for i in et:
                        temp.append(i.lemma_)
                    event_l = '_'.join(temp)
                    tag1 = find_concept(event_l, vocab_path)
                    if not tag1:
                        tag2 = find_concept(e_text, vocab_path)
                        if not tag2:
                            for i in [et[0], et[-1]]:
                                if i.pos_ == 'NOUN' and len(i.lemma_) > 2 and find_concept(i.lemma_, vocab_path):
                                    event_other = i.lemma_
                                    break
                            if event_other is None:
                                for i in [et[0], et[-1]]:
                                    if i.pos_ == 'VERB' and len(i.lemma_) > 2 and find_concept(i.lemma_, vocab_path):
                                        event_other = i.lemma_
                                        break
                if tag1:
                    event['concept'] = event_l
                elif tag2:
                    event['concept'] = e_text
                    print(e_text, event_l)
                else:
                    event['concept'] = event_other
                    print(e_text, event_l, event_other)
        docs.append(j)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for doc in docs:
            fout.write(json.dumps(doc) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print(len(docs))
    print()


if __name__ == "__main__":
    ground("../../data/story/story_event.jsonl", "../../data/cpnet/concept.txt", "../../data/story/ground_res.jsonl")
    #
    # s = "a revolving door is convenient for two direction travel, but it also serves as a security measure at a bank."
    # a = "bank"
    # nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    # nlp.add_pipe('sentencizer')
    # ans_words = nlp(a)
    # doc = nlp(s)
    # for i in doc:
    #     print(i)
    # ans_matcher = Matcher(nlp.vocab)
    # print([{'TEXT': token.text.lower()} for token in ans_words])
    # ans_matcher.add("ok", [[{'TEXT': token.text.lower()} for token in ans_words]])
    #
    # matches = ans_matcher(doc)
    # for a, b, c, in matches:
    #     print(doc[b:c])
    #     print(a, b, c)
