from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate_model(y_true, anomaly_scores):
    roc = roc_auc_score(y_true, anomaly_scores)
    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
    pr_auc = auc(recall, precision)
    return {
        "roc_auc": roc,
        "pr_auc": pr_auc
    }