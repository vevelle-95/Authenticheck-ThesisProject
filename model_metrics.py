"""Fixed-label metrics for our own model, independent of comparison services."""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(targets, predictions, labels):
    if len(targets) == 0:
        return {"count": 0, "accuracy": None, "macro_precision": None, "macro_recall": None, "macro_f1": None}
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=list(labels), zero_division=0,
    )
    return {
        "count": len(targets), "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "support": support.astype(int).tolist(),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(labels)).tolist(),
    }
