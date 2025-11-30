import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def validate_encoder_outputs():
    """Validate encoder outputs and check for issues"""
    print("=== ENCODER VALIDATION ===")
    
    # Load latent features
    lstm_dir = Path("e:/electricity_theft/data/processed/lstm")
    latent_features = np.load(lstm_dir / "latent_features.npy")
    
    print(f"Latent shape: {latent_features.shape}")
    print(f"Data type: {latent_features.dtype}")
    
    # Check for NaNs
    nan_count = np.isnan(latent_features).sum()
    print(f"NaN values: {nan_count}")
    
    if nan_count > 0:
        print("[ERROR] Found NaN values in latent features!")
        return False
    
    # Check variance across dimensions
    variances = np.var(latent_features, axis=0)
    print(f"Variance per dimension: {variances}")
    print(f"Min variance: {variances.min():.6f}")
    print(f"Max variance: {variances.max():.6f}")
    
    # Check if latent is flat (very low variance)
    if variances.max() < 1e-6:
        print("[ERROR] Latent features are flat - AE training failed!")
        return False
    
    # PCA plot
    if latent_features.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_features)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=20)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA of Latent Features')
        plt.grid(True, alpha=0.3)
        plt.savefig('e:/electricity_theft/models/trained_models/latent_pca.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"PCA explained variance: {pca.explained_variance_ratio_[:2]}")
    
    return True

def retrain_oneclass_svm():
    """Retrain OneClassSVM with tuned parameters"""
    print("\n=== RETRAINING ONECLASS SVM ===")
    
    lstm_dir = Path("e:/electricity_theft/data/processed/lstm")
    models_dir = Path("e:/electricity_theft/models")
    
    # Load and prepare data
    latent_features = np.load(lstm_dir / "latent_features.npy")
    
    # Remove NaNs if any
    if np.isnan(latent_features).any():
        print("Removing NaN values...")
        latent_features = latent_features[~np.isnan(latent_features).any(axis=1)]
    
    print(f"Training on {latent_features.shape[0]} samples, {latent_features.shape[1]} features")
    
    # Scale features
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_features)
    
    # Train OneClassSVM with tuned parameters
    print("Training OneClassSVM with tuned parameters...")
    svdd_model = OneClassSVM(
        kernel="rbf",
        gamma=0.01,
        nu=0.05,
        random_state=42
    )
    
    svdd_model.fit(latent_scaled)
    
    # Generate scores and labels
    svdd_scores = svdd_model.decision_function(latent_scaled)
    
    # Check score distribution
    print(f"Score statistics:")
    print(f"  Min: {svdd_scores.min():.6f}")
    print(f"  Max: {svdd_scores.max():.6f}")
    print(f"  Mean: {svdd_scores.mean():.6f}")
    print(f"  Std: {svdd_scores.std():.6f}")
    print(f"  Unique values: {len(np.unique(svdd_scores))}")
    
    if svdd_scores.std() < 1e-6:
        print("[ERROR] Score distribution is still constant!")
        return False
    
    # Generate labels (5% as anomalies)
    threshold = np.percentile(svdd_scores, 5)
    svdd_labels = (svdd_scores < threshold).astype(int)
    
    print(f"Generated {np.sum(svdd_labels)} anomalies ({np.mean(svdd_labels):.1%})")
    
    # Save results
    np.save(lstm_dir / "svdd_scores.npy", svdd_scores)
    np.save(lstm_dir / "svdd_labels.npy", svdd_labels)
    joblib.dump(svdd_model, models_dir / "svdd_oneclass.pkl")
    joblib.dump(scaler, models_dir / "svdd_scaler.pkl")
    
    print("\n[SAVED] svdd_scores.npy")
    print("[SAVED] svdd_labels.npy") 
    print("[SAVED] svdd_oneclass.pkl")
    print("[SAVED] svdd_scaler.pkl")
    
    # Plot score distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(svdd_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    plt.xlabel('SVDD Score')
    plt.ylabel('Frequency')
    plt.title('SVDD Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    normal_scores = svdd_scores[svdd_labels == 0]
    anomaly_scores = svdd_scores[svdd_labels == 1]
    
    plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red')
    plt.xlabel('SVDD Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('e:/electricity_theft/models/trained_models/svdd_score_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    """Main validation and retraining pipeline"""
    print("SVDD VALIDATION AND RETRAINING PIPELINE")
    print("=" * 50)
    
    # Step 1: Validate encoder outputs
    if not validate_encoder_outputs():
        print("\n[FAILED] Encoder validation failed - need to retrain autoencoder")
        return
    
    print("\n[PASSED] Encoder validation successful")
    
    # Step 2: Retrain OneClassSVM
    if not retrain_oneclass_svm():
        print("\n[FAILED] SVDD retraining failed")
        return
    
    print("\n[SUCCESS] SVDD retraining completed successfully!")
    print("You can now run day13_svdd_evaluation.py again")

if __name__ == "__main__":
    main()