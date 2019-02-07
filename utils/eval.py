import argparse
import os
import sys
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import math
from tqdm import tqdm
import numpy as np

def div(x, y):
    try:
        return float(x) / float(y)
    except:
        return 0

def get_values(path):
    y_true = []
    y_pred = []
    with open(path) as f:
        for line in f:
            try:
                _, _ , t, p = line.rstrip().split('\t')
            except:
                t, p = line.rstrip().split('\t')
            y_true.append(float(t))
            y_pred.append(float(p))
    return y_true, y_pred

def wbless(y_true, y_pred):
    print('tau', 'accuracy', 'precision', 'recall', 'f1')
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    for i, (p, r, t) in enumerate(zip(precision, recall, thresholds)):
        f1 = div(2 * p * r, p + r)
        y_pred_int = []
        for q in y_pred:
            if q >= t:
                y_pred_int.append(1)
            else:
                y_pred_int.append(0)
        print(t, accuracy_score(y_true, y_pred_int), p, r, f1)

def hyperlex(y_true, y_pred):
    print('Spearmanr', 'MAE', 'RMSE')
    print(spearmanr(y_true, y_pred).correlation, mean_absolute_error(y_true, y_pred), math.sqrt(mean_squared_error(y_true, y_pred)))

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
    path = sys.argv[1]
    y_true, y_pred = get_values(path)
    method = os.path.normpath(path).split('/')[2]
    if method == 'wbless':
        wbless(y_true, y_pred)
    elif method == 'hyperlex':
        hyperlex(y_true, y_pred)
    elif method == 'categorization':
        categorization(y_true, y_pred)
    else:
        print('Invalid method: ' + method)
