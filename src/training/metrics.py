"""
Utility functions to compute common classification metrics.

We will calculate:
- accuracy
- precision
- recall
- F1-score
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred):
    """
    y_true: list/array of true labels (0/1)
    y_pred: list/array of predicted labels (0/1)
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }