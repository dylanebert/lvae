import argparse
import os
import sys
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import math
from tqdm import tqdm
import numpy as np
from collections import defaultdict

def div(x, y):
    try:
        return float(x) / float(y)
    except:
        return 0

def get_values(path, test_path):
    vals = defaultdict(list)
    with open(path) as f:
        for line in f:
            w1, w2, l1, l2 , t, p = line.rstrip().split('\t')
            vals[(w1, w2, l1, l2)] += [float(t), float(p)]
    with open(test_path) as f:
        for line in f:
            w1, w2, l1, l2 , t, p = line.rstrip().split('\t')
            vals[(w1, w2, l1, l2)] += [float(t), float(p)]
    for v in vals.values():
        assert(len(v) == 3)
    return vals

def get_categorization_vals(path):
    y_true, y_pred = [], []
    with open(path) as f:
        for line in f:
            l, t, p = line.rstrip().split('\t')
            y_true.append(float(t))
            y_pred.append(float(p))
    return y_true, y_pred

def wbless(vals):
    print('tau', 'accuracy', 'precision', 'recall', 'f1')
    y_true = []
    y_pred = []
    y_pred_test = []
    for key, val in vals.items():
        y_true.append(val[0])
        y_pred.append(val[1])
        y_pred_test.append(val[2])
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    max_f1 = 0
    max_f1_idx = None
    for i, (p, r, t) in enumerate(zip(precision, recall, thresholds)):
        f1 = div(2 * p * r, p + r)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_idx = i
    tau = thresholds[max_f1_idx]
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for t, p in zip(y_true, y_pred_test):
        if p >= tau:
            if t == 1:
                tp += 1
            else:
                fp += 1
        else:
            if t == 1:
                fn += 1
            else:
                tn += 1
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / float(tp + tn + fp + fn)
    print('\t'.join([str(i) for i in [tau, acc, precision, recall, f1]]))

def hyperlex(values):
    print('Spearmanr', 'MAE', 'RMSE')
    y_true = []
    y_pred = []
    y_pred_test = []
    for key, val in vals.items():
        y_true.append(val[0])
        y_pred.append(val[1])
        y_pred_test.append(val[2])
    print(spearmanr(y_true, y_pred_test).correlation, mean_absolute_error(y_true, y_pred_test), math.sqrt(mean_squared_error(y_true, y_pred_test)))

def categorization_discrete(y_true, y_pred):
    print('accuracy', 'precision', 'recall', 'f1')
    tp, fp, fn, tn = [0] * 4
    for t, p in zip(y_true, y_pred):
        if p == 1:
            if t == 1:
                tp += 1
            else:
                fp += 1
        else:
            if t == 1:
                fn += 1
            else:
                tn += 1
    p = div(tp, tp + fp)
    r = div(tp, tp + fn)
    f1 = div(2 * p * r, p + r)
    a = div(tp + tn, tp + tn + fp + fn)
    print(a, p, r, f1)

def categorization_continuous(y_true, y_pred):
    print('tau', 'accuracy', 'precision', 'recall', 'f1')
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    max_f1_idx = None
    max_f1 = 0
    for i, (p, r, t) in enumerate(list(zip(precision, recall, thresholds))):
        f1 = div(2 * p * r, p + r)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_idx = i

    y_pred_int = []
    for q in y_pred:
        if q >= thresholds[max_f1_idx]:
            y_pred_int.append(1)
        else:
            y_pred_int.append(0)
    print(thresholds[max_f1_idx], accuracy_score(y_true, y_pred_int), precision[max_f1_idx], recall[max_f1_idx], max_f1)

def categorization(y_true, y_pred):
    y_true = [int(i) for i in y_true]
    discrete = all(i.is_integer() for i in y_pred)
    if discrete:
        categorization_discrete(y_true, y_pred)
    else:
        categorization_continuous(y_true, y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', type=int, default=2)
    parser.add_argument('--method', help='classical or prototype', type=str, required=True)
    parser.add_argument('--dset', help='wbless or hyperlex or categorization', type=str, required=True)
    parser.add_argument('--reduced', action='store_true')
    args = parser.parse_args()

    path = 'results/' + str(args.latent_size) + '/' + args.dset + '/' + args.method
    if args.reduced:
        path += '_2d'
    if args.dset == 'categorization':
        path += '.txt'
        y_true, y_pred = get_categorization_vals(path)
        categorization(y_true, y_pred)
    else:
        test_path = path + '_test.txt'
        path += '_dev.txt'
        vals = get_values(path, test_path)
        if args.dset == 'wbless':
            wbless(vals)
        elif args.dset == 'hyperlex':
            hyperlex(vals)
