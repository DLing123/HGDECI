#!/usr/tensorflow/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2020/7/20 10:28
# @Author   : Dling
# @FileName : eval_results.py
# @Software : PyCharm
# @email    : dling@tongji.edu.cn
import torch
import numpy as np
from utils import *


def compute_f1(preds, labels):
    n_gold = n_pred = n_correct = 0
    c_acc = np.array([])
    for pred, label in zip(preds, labels):
        pred_ = pred.reshape(-1)
        label_ = label.reshape(-1)
        c_acc = np.append(c_acc, (pred_ == label_))
        for p, l in zip(pred_, label_):
            if p != 0:
                n_pred += 1
            if l != 0:
                n_gold += 1
            if (p != 0) and (l != 0) and (p == l):
                n_correct += 1
    if n_correct == 0:
        prec, recall, f1, all_recall, all_f1 = 0., 0., 0., 0., 0.
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0

    acc = c_acc.mean()
    print_f1(prec, recall, f1, n_correct, n_pred, n_gold, acc)
    print()
    return f1


def evaluate(model, batches):
    model.eval()

    all_label_ids = None
    all_predict_ids = None

    for batch in batches:
        with torch.no_grad():
            logits, pred_tri = model(batch, training=False)

        label_ids = batch[1].detach().cpu().numpy()
        pred_ids = pred_tri.detach().cpu().numpy()
        assert pred_ids.shape == label_ids.shape  # bs, 1

        if all_label_ids is None:
            all_label_ids = label_ids
            all_predict_ids = pred_ids
        else:
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)
            all_predict_ids = np.append(all_predict_ids, pred_ids, axis=0)
    assert all_label_ids.shape == all_predict_ids.shape
    f1 = compute_f1(all_predict_ids, all_label_ids)
    return f1
