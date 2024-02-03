import pdb
import json
import logging.handlers
import argparse
import os
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from networkx import DiGraph, relabel_nodes, all_pairs_shortest_path_length
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, h_recall_score, h_precision_score, \
    fill_ancestors, multi_labeled
import sys
sys.path.append('.')


KEYS = ['id','labels']

#F1
def _h_fbeta_score(y_true, y_pred, class_hierarchy, beta=1., root=ROOT):
    hP = _h_precision_score(y_true, y_pred, class_hierarchy, root=root)
    hR = _h_recall_score(y_true, y_pred, class_hierarchy, root=root)
    return (1. + beta ** 2.) * hP * hR / (beta ** 2. * hP + hR)

def _fill_ancestors(y, graph, root, copy=True):
    y_ = y.copy() if copy else y
    paths = all_pairs_shortest_path_length(graph.reverse(copy=False))
    for target, distances in paths:
        if target == root:
            continue
        ix_rows = np.where(y[:, target] > 0)[0]
        ancestors = list(filter(lambda x: x != ROOT,distances.keys()))
        y_[tuple(np.meshgrid(ix_rows, ancestors))] = 1
    graph.reverse(copy=False)
    return y_
def _h_recall_score(y_true, y_pred, class_hierarchy, root=ROOT):
    y_true_ = _fill_ancestors(y_true, graph=class_hierarchy, root=root)
    y_pred_ = _fill_ancestors(y_pred, graph=class_hierarchy, root=root)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_positives = np.count_nonzero(y_true_)

    return true_positives / all_positives

def _h_precision_score(y_true, y_pred, class_hierarchy, root=ROOT):
    y_true_ = _fill_ancestors(y_true, graph=class_hierarchy, root=root)
    y_pred_ = _fill_ancestors(y_pred, graph=class_hierarchy, root=root)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_results = np.count_nonzero(y_pred_)

    return true_positives / all_results



def _read_gold_and_pred(pred_fpath, gold_fpath):

  gold_labels = {}
  with open(gold_fpath, encoding='utf-8') as gold_f:
    gold = json.load(gold_f)
    for obj in gold:
      gold_labels[obj['id']] = obj['labels']

  pred_labels = {}
  with open(pred_fpath, encoding='utf-8') as pred_f:
    pred = json.load(pred_f)
    for obj in pred:
      pred_labels[obj['id']] = obj['labels']

  return pred_labels, gold_labels

def evaluate_h(pred_fpath, gold_fpath, G):
    pred_labels, gold_labels = _read_gold_and_pred(pred_fpath, gold_fpath)

    gold = []
    pred = []
    for id in gold_labels:
        gold.append(gold_labels[id])
        pred.append(pred_labels[id])
    with multi_labeled(gold, pred, G) as (gold_, pred_, graph_):
        return  _h_precision_score(gold_, pred_,graph_), _h_recall_score(gold_, pred_,graph_), _h_fbeta_score(gold_, pred_,graph_)
    

