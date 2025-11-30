import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

def validate_encoder_outputs():
    """Validate encoder outputs for NaNs, variance, and flatness"""
    
    latent_path = "data/processed/lstm/latent_features.npy"
    if not os.path.exists(latent_path):
        print(f"ERROR: Latent features file not found: {latent_path}")
        return False
        
    latent_features = np.load(latent_path)
    print(f"Latent features shape: {latent_features.shape}")
    
    # Check for NaNs
    nan_count = np.isnan(latent_features).sum()
    print(f"NaN values found: {nan_count}")
    if nan_count > 0:
        print("CRITICAL: NaN values detected in latent features!")
        return False
    
    # Check variance across dimensions
    variances = np.var(latent_features, axis=0)
    print(f"Variance per dimension: {variances}")
    print(f"Min variance: {variances.min():.6f}")
    print(f"Max variance: {variances.max():.6f}")
    
    # Check if latent looks flat (very low variance)
    if variances.max() < 1e-6:
        print("CRITICAL: Latent space appears flat - AE training failed!")
        return False
    
    # PCA plot on latents
    if latent_features.shape[1] >= 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_features)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=1)
        plt.title('PCA Plot of Latent Features')
        plt.xlabel(f'PC1 (var: {pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 (var: {pca.explained_variance_ratio_[1]:.3f})')
        os.makedirs('models/trained_models', exist_ok=True)
        plt.savefig('models/trained_models/latent_pca_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    print("SUCCESS: Encoder outputs validation passed!")
    return True

def retrain_oneclass_svm():
    """Retrain OneClassSVM with tuned parameters"""
    
    latent_features = np.load("data/processed/lstm/latent_features.npy")
    
    # Scale features
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_features)
    
    # Train OneClassSVM with specified parameters
    print("Training OneClassSVM with tuned parameters...")
    oneclass_svm = OneClassSVM(kernel="rbf", gamma=0.01, nu=0.05)
    oneclass_svm.fit(latent_scaled)
    
    # Get scores
    scores = oneclass_svm.decision_function(latent_scaled)
    labels = oneclass_svm.predict(latent_scaled)
    
    # Check score distribution
    print(f"Score statistics:")
    print(f"  Mean: {scores.mean():.6f}")
    print(f"  Std: {scores.std():.6f}")
    print(f"  Min: {scores.min():.6f}")
    print(f"  Max: {scores.max():.6f}")
    
    # Check if distribution is constant
    if scores.std() < 1e-6:
        print("CRITICAL: Score distribution is constant!")
        return False
    
    # Plot score distribution
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
    plt.title('SVDD Score Distribution')
    plt.xlabel('Decision Function Score')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.plot(scores[:1000])
    plt.title('SVDD Scores Timeline (First 1000)')
    plt.xlabel('Sample Index')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('models/trained_models/svdd_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save new fixed scores, labels, and models
    print("Saving updated models and results...")
    np.save("data/processed/lstm/svdd_scores.npy", scores)
    np.save("data/processed/lstm/svdd_labels.npy", labels)
    
    os.makedirs('models', exist_ok=True)
    with open("models/svdd_oneclass.pkl", "wb") as f:
        pickle.dump(oneclass_svm, f)
    
    with open("models/svdd_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print("SUCCESS: OneClassSVM retrained and saved successfully!")
    return True

def main():
    """Main validation and retraining pipeline"""
    
    print("=== ENCODER OUTPUT VALIDATION ===")
    encoder_valid = validate_encoder_outputs()
    
    if not encoder_valid:
        print("ERROR: Encoder validation failed. Please retrain the autoencoder first.")
        return
    
    print("\n=== ONECLASS SVM RETRAINING ===")
    svm_success = retrain_oneclass_svm()
    
    if svm_success:
        print("\nSUCCESS: All validation and retraining completed successfully!")
        print("Files saved:")
        print("  - data/processed/lstm/svdd_scores.npy")
        print("  - data/processed/lstm/svdd_labels.npy") 
        print("  - models/svdd_oneclass.pkl")
        print("  - models/svdd_scaler.pkl")
    else:
        print("\nERROR: SVM retraining failed.")

if __name__ == "__main__":
    main()