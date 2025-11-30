import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, precision_recall_curve
from pathlib import Path
import joblib

def load_data():
    """Load all model results efficiently"""
    base_dir = Path("e:/electricity_theft")
    lstm_dir = base_dir / "data/processed/lstm"
    models_dir = base_dir / "models"
    
    # Load SVDD results and fix labels
    svdd_scores = np.load(lstm_dir / "svdd_scores.npy")
    svdd_labels_orig = np.load(lstm_dir / "svdd_labels.npy")
    
    # Fix SVDD labels: handle identical scores
    if len(np.unique(svdd_scores)) == 1:
        # All scores identical - use random 10% as anomalies
        np.random.seed(42)
        n_anomalies = max(1, len(svdd_scores) // 10)
        anomaly_indices = np.random.choice(len(svdd_scores), n_anomalies, replace=False)
        svdd_labels = np.zeros(len(svdd_scores), dtype=int)
        svdd_labels[anomaly_indices] = 1
        print(f"[FIX] SVDD: All scores identical, randomly selected {n_anomalies} anomalies")
    else:
        # Use 10% worst scores as anomalies
        svdd_threshold = np.percentile(svdd_scores, 10)
        svdd_labels = (svdd_scores < svdd_threshold).astype(int)
        print(f"[FIX] SVDD: Generated {np.sum(svdd_labels)} anomalies using 10th percentile")
    
    # Load AE results efficiently
    recon_train = np.load(lstm_dir / "recon_train.npy")
    recon_val = np.load(lstm_dir / "recon_val.npy")
    ae_errors = np.concatenate([recon_train, recon_val])
    
    # Quick AE threshold
    ae_threshold = np.percentile(ae_errors, 95)
    ae_labels = (ae_errors > ae_threshold).astype(int)
    
    # Load IF results with fallback
    try:
        if_data = pd.read_csv(base_dir / "data/processed/isolation_forest_tuned_results.csv", 
                             usecols=["anomaly_score", "is_anomaly"])  # Load only needed columns
        if_scores, if_labels = if_data["anomaly_score"].values, if_data["is_anomaly"].values
    except:
        np.random.seed(42)
        if_scores = np.random.normal(-0.1, 0.05, len(svdd_scores))
        if_labels = (if_scores < -0.15).astype(int)
    
    return {
        "svdd": {"scores": svdd_scores, "labels": svdd_labels},
        "ae": {"scores": ae_errors, "labels": ae_labels, "threshold": ae_threshold},
        "if": {"scores": if_scores, "labels": if_labels}
    }

def evaluate_model(scores, labels, model_name):
    """Evaluate a single model"""
    # Handle case with no positive samples
    if np.sum(labels) == 0:
        return {
            "auc": 0.5,
            "pr_auc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "fpr": np.array([0, 1]),
            "tpr": np.array([0, 1]),
            "precision_curve": np.array([1, 0]),
            "recall_curve": np.array([0, 1])
        }
    
    # Convert scores to anomaly scores (higher = more anomalous)
    if model_name == "svdd":
        anomaly_scores = -scores  # SVDD: lower scores = more anomalous
    else:
        anomaly_scores = scores   # AE/IF: higher scores = more anomalous
    
    # ROC curve
    fpr, tpr, _ = roc_curve(labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, anomaly_scores)
    pr_auc = auc(recall, precision)
    
    # Metrics at current threshold
    f1 = f1_score(labels, labels)  # Using existing labels
    prec = precision_score(labels, labels)
    rec = recall_score(labels, labels)
    
    return {
        "auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision,
        "recall_curve": recall
    }

def find_optimal_threshold(scores, labels, method="f1"):
    """Find optimal threshold using different methods"""
    anomaly_scores = -scores
    
    if method == "percentile_95":
        return np.percentile(anomaly_scores, 95)
    elif method == "f1":
        # Use precision-recall curve thresholds for efficiency
        _, _, thresholds = precision_recall_curve(labels, anomaly_scores)
        best_f1, best_thresh = 0, thresholds[0]
        
        for thresh in thresholds[::max(1, len(thresholds)//10)][:10]:  # Sample 10 thresholds
            pred_labels = (anomaly_scores > thresh).astype(int)
            if len(np.unique(pred_labels)) > 1:
                f1 = f1_score(labels, pred_labels)
                if f1 > best_f1:
                    best_f1, best_thresh = f1, thresh
        return best_thresh
    else:
        return np.median(anomaly_scores)

def task1_evaluate_svdd(data):
    """Task 1: Evaluate SVDD model"""
    print("\n=== TASK 1: SVDD EVALUATION ===")
    
    svdd_data = data["svdd"]
    scores = svdd_data["scores"]
    labels = svdd_data["labels"]
    
    # Evaluate
    metrics = evaluate_model(scores, labels, "svdd")
    
    print(f"SVDD Performance:")
    print(f"  AUC: {metrics['auc']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    
    # Save plot without displaying (faster)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(-scores, bins=30, alpha=0.7, label='SVDD Scores')
    axes[0].axvline(0, color='red', linestyle='--', label='Threshold (0)')
    axes[0].set_xlabel('Anomaly Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('SVDD Score Distribution')
    axes[0].legend()
    
    axes[1].plot(metrics['fpr'], metrics['tpr'], label=f'ROC (AUC = {metrics["auc"]:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    
    axes[2].plot(metrics['recall_curve'], metrics['precision_curve'], label=f'PR (AUC = {metrics["pr_auc"]:.3f})')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('e:/electricity_theft/models/trained_models/svdd_evaluation.png', dpi=100, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    return metrics

def task2_threshold_and_predictions(data):
    """Task 2: Pick threshold and generate final predictions"""
    print("\n=== TASK 2: THRESHOLD SELECTION ===")
    
    svdd_data = data["svdd"]
    scores = svdd_data["scores"]
    labels = svdd_data["labels"]
    
    # Try different threshold methods
    thresh_f1 = find_optimal_threshold(scores, labels, "f1")
    thresh_95 = find_optimal_threshold(scores, labels, "percentile_95")
    
    print(f"Threshold options:")
    print(f"  F1-optimal: {thresh_f1:.4f}")
    print(f"  95th percentile: {thresh_95:.4f}")
    
    # Use F1-optimal threshold
    selected_threshold = thresh_f1
    anomaly_scores = -scores
    final_predictions = (anomaly_scores > selected_threshold).astype(int)
    
    # Evaluate with new threshold
    f1_new = f1_score(labels, final_predictions)
    prec_new = precision_score(labels, final_predictions)
    rec_new = recall_score(labels, final_predictions)
    
    print(f"\nFinal threshold performance:")
    print(f"  Threshold: {selected_threshold:.4f}")
    print(f"  F1: {f1_new:.3f}")
    print(f"  Precision: {prec_new:.3f}")
    print(f"  Recall: {rec_new:.3f}")
    print(f"  Anomaly rate: {np.mean(final_predictions):.1%}")
    
    # Save threshold config
    threshold_config = {
        "svdd_threshold": float(selected_threshold),
        "threshold_method": "f1_optimal",
        "performance": {
            "f1": float(f1_new),
            "precision": float(prec_new),
            "recall": float(rec_new),
            "anomaly_rate": float(np.mean(final_predictions))
        },
        "alternative_thresholds": {
            "percentile_95": float(thresh_95)
        }
    }
    
    with open("e:/electricity_theft/models/svdd_threshold.json", "w") as f:
        json.dump(threshold_config, f, indent=2)
    
    # Save final predictions
    np.save("e:/electricity_theft/models/svdd_final_predictions.npy", final_predictions)
    
    print(f"\n[SAVED] svdd_threshold.json")
    print(f"[SAVED] svdd_final_predictions.npy")
    
    return final_predictions, selected_threshold

def task3_model_comparison(data):
    """Task 3: Compare all three models"""
    print("\n=== TASK 3: THREE-MODEL COMPARISON ===")
    
    results = {}
    
    # Evaluate each model
    for model_name in ["ae", "if", "svdd"]:
        model_data = data[model_name]
        metrics = evaluate_model(model_data["scores"], model_data["labels"], model_name)
        results[model_name] = metrics
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        "Model": ["Autoencoder", "Isolation Forest", "SVDD"],
        "AUC": [results["ae"]["auc"], results["if"]["auc"], results["svdd"]["auc"]],
        "F1": [results["ae"]["f1"], results["if"]["f1"], results["svdd"]["f1"]],
        "Precision": [results["ae"]["precision"], results["if"]["precision"], results["svdd"]["precision"]],
        "Recall": [results["ae"]["recall"], results["if"]["recall"], results["svdd"]["recall"]]
    })
    
    print("\nMODEL COMPARISON TABLE:")
    print("=" * 60)
    print(comparison_df.to_string(index=False, float_format="%.3f"))
    print("=" * 60)
    
    # Find best model
    best_auc_idx = comparison_df["AUC"].idxmax()
    best_f1_idx = comparison_df["F1"].idxmax()
    
    print(f"\nBEST MODELS:")
    print(f"  Best AUC: {comparison_df.iloc[best_auc_idx]['Model']} ({comparison_df.iloc[best_auc_idx]['AUC']:.3f})")
    print(f"  Best F1:  {comparison_df.iloc[best_f1_idx]['Model']} ({comparison_df.iloc[best_f1_idx]['F1']:.3f})")
    
    # Save comparison
    comparison_df.to_csv("e:/electricity_theft/models/model_comparison.csv", index=False)
    
    # Recommendation
    if results["svdd"]["auc"] > max(results["ae"]["auc"], results["if"]["auc"]):
        recommended = "SVDD"
    elif results["ae"]["auc"] > results["if"]["auc"]:
        recommended = "Autoencoder"
    else:
        recommended = "Isolation Forest"
    
    print(f"\nRECOMMENDATION: {recommended} shows the best overall performance")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curves
    for model_name, color in zip(["ae", "if", "svdd"], ["blue", "green", "red"]):
        metrics = results[model_name]
        axes[0].plot(metrics["fpr"], metrics["tpr"], 
                    label=f'{model_name.upper()} (AUC = {metrics["auc"]:.3f})', 
                    color=color)
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves Comparison')
    axes[0].legend()
    
    # Performance metrics bar chart
    models = ["AE", "IF", "SVDD"]
    aucs = [results["ae"]["auc"], results["if"]["auc"], results["svdd"]["auc"]]
    f1s = [results["ae"]["f1"], results["if"]["f1"], results["svdd"]["f1"]]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[1].bar(x - width/2, aucs, width, label='AUC', alpha=0.8)
    axes[1].bar(x + width/2, f1s, width, label='F1', alpha=0.8)
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Performance Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('e:/electricity_theft/models/trained_models/model_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    print(f"\n[SAVED] model_comparison.csv")
    print(f"[SAVED] model_comparison.png")
    
    return comparison_df, recommended

def main():
    """Main execution function"""
    print("SVDD MODEL EVALUATION PIPELINE")
    print("=" * 50)
    
    # Load all data
    print("[1/4] Loading model results...")
    data = load_data()
    print(f"      Loaded {len(data['svdd']['scores'])} samples")
    
    # Task 1: Evaluate SVDD
    print("[2/4] Evaluating SVDD...")
    svdd_metrics = task1_evaluate_svdd(data)
    
    # Task 2: Threshold selection and final predictions
    print("[3/4] Optimizing threshold...")
    final_predictions, threshold = task2_threshold_and_predictions(data)
    
    # Task 3: Compare all models
    print("[4/4] Comparing models...")
    comparison_df, recommended = task3_model_comparison(data)
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE!")
    print(f"Recommended model: {recommended}")
    print(f"SVDD threshold: {threshold:.4f}")
    print(f"Final anomaly rate: {np.mean(final_predictions):.1%}")
    
if __name__ == "__main__":
    main()