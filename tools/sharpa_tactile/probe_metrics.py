"""Shared metrics and CSV output for independent annotation probes."""
import csv
from pathlib import Path
import numpy as np


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(y, probability):
    y = np.asarray(y, int)
    p = np.asarray(probability)
    pred = p >= .5
    cm = np.array([[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)])
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(2), where=cm.sum(1)>0)
    precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(2), where=cm.sum(0)>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(2), where=precision+recall>0)
    # Tied scores are processed as one group. PR-AUC uses average precision.
    order = np.argsort(-p, kind='stable')
    sorted_y, sorted_p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(y)-1]
    tp = np.cumsum(sorted_y)[ends].astype(float)
    fp = (ends+1)-tp
    positives, negatives = int(y.sum()), int((y == 0).sum())
    pr_auc = roc_auc = None
    if positives:
        rec = tp/positives
        pr_auc = float(np.sum(np.diff(np.r_[0, rec]) * tp/(tp+fp)))
        if negatives:
            roc_auc = float(np.trapz(np.r_[0, rec], np.r_[0, fp/negatives]))
    return dict(accuracy=float((pred == y).mean()), balanced_accuracy=float(recall.mean()),
                precision=float(precision[1]), recall=float(recall[1]), f1=float(f1[1]),
                macro_f1=float(f1.mean()), fpr=float(recall[0]*-1+1),
                pr_auc=pr_auc, roc_auc=roc_auc, confusion_matrix=cm.tolist(),
                positive=positives, negative=negatives)
