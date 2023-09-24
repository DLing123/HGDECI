#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2020/11/25 17:04
# @Author   : Dling
# @FileName : embedding.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import torch
import torch.nn as nn
import codecs
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math


class WordVecMat:
    def __init__(self, file_name, word2id, dim=50):
        self.embeddings = []
        self.load_embedding(file_name, word2id, dim)

    def load_embedding(self, file_name, word2id, dim):
        word2vec = {}
        for line in codecs.open(file_name, 'r', encoding='utf-8'):
            line = line.strip().split()
            vec = np.array([float(i) for i in line[1:]])
            vec = vec / np.sqrt(np.sum(vec * vec))  # 平方求和,和开平方，作除数
            if len(line) == dim + 1:
                word2vec[line[0]] = vec
            else:
                print("invalid embedding")

        for word in word2id:
            if word in word2vec:
                self.embeddings.append(word2vec[word])
            elif word == "<PADDING>":
                self.embeddings.append(self.get_zero_vector(dim))
            else:
                self.embeddings.append(self.get_init_vector(dim))

    def get_init_vector(self, dim):
        # scale = np.sqrt(6. / (1 + dim))
        scale = 0.1
        vec = np.random.uniform(low=-scale, high=scale, size=[dim])
        vec = vec / np.sqrt(np.sum(vec * vec))
        assert abs(np.sum(vec * vec) - 1.0) < 0.1
        return list(vec)

    def get_zero_vector(self, dim):
        return [0.0] * dim

    def get_embeddings(self):
        return np.array(self.embeddings)


class Embedding(nn.Module):

    def __init__(self, confs):
        nn.Module.__init__(self)
        self.max_length = confs["max_length"]
        self.word_embedding_dim = confs["word_embedding_dim"]
        self.pos_embedding_dim = confs["pos_embedding_dim"]

        wordvecmat = WordVecMat(confs["file_name"], confs["word2id"])
        self.embedding = wordvecmat.get_embeddings()

        word_vec_mat = torch.from_numpy(self.embeddings)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim, padding_idx=word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data.copy_(word_vec_mat)

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * self.max_length, self.pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * self.max_length, self.pos_embedding_dim, padding_idx=0)

    def forward(self, inputs):
        word = inputs['word']
        pos1 = inputs['pos1']
        pos2 = inputs['pos2']

        x = torch.cat([self.word_embedding(word),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2)
        return x






