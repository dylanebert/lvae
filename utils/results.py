import pandas as pd
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def div(x, y):
    try:
        return float(x) / float(y)
    except:
        return 0

def accuracy_report(df):
    precision, recall, thresholds = precision_recall_curve(df['true'].values, df['predicted'].values)
    f1 = [div(2*p*r, p+r) for p, r in zip(precision, recall)]
    best_idx = np.argmax(f1)
    sum, correct = 0, 0
    for t, p in zip(df['true'].values, df['predicted'].values):
        if p >= thresholds[best_idx]:
            if t == 1:
                correct += 1
        else:
            if t == 0:
                correct += 1
        sum += 1
    results = pd.DataFrame(data={'accuracy': [correct / float(sum)], 'precision': [precision[best_idx]], 'recall': [recall[best_idx]], 'f1': [f1[best_idx]]})
    print(results)
    plt.plot(recall, precision)
    plt.show()

def get_df(path):
    df = pd.read_csv(path, sep='\t', names=['w1', 'w2', 'true', 'predicted'], header=None)
    return df

model = sys.argv[1]
path = os.path.join('results', model, 'classical_wbless_outliers.txt')
df = get_df(path)
accuracy_report(df)
