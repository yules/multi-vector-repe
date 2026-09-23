import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

print("Loading cached vectors...")
data = np.load("./data/latent_vectors.npz", allow_pickle=True)
X_benign = data['X_benign']
X_malicious = data['X_malicious']
prompts_benign = data['prompts_benign']
prompts_malicious = data['prompts_malicious']

# 1. Dimensionality Reduction
# Compressing 3,072 dimensions into the top 150 principal concepts 
# ensures the covariance matrix is strictly invertible.
pca = PCA(n_components=150, random_state=42)
X_benign_pca = pca.fit_transform(X_benign)
X_malicious_pca = pca.transform(X_malicious)

# 2. Fit the Mahalanobis Covariance Estimator
print("Calculating Mahalanobis covariance manifold...")
# LedoitWolf is a robust covariance estimator ideal for high-dimensional data
cov_estimator = LedoitWolf().fit(X_benign_pca)

# Calculate squared Mahalanobis distances
dist_benign = cov_estimator.mahalanobis(X_benign_pca)
dist_malicious = cov_estimator.mahalanobis(X_malicious_pca)

# 3. Set the Anomaly Threshold
# We tolerate a 5% false positive rate on normal traffic to establish the boundary
threshold = np.percentile(dist_benign, 95)

# 4. Evaluate
preds_benign = np.where(dist_benign > threshold, -1, 1)
preds_malicious = np.where(dist_malicious > threshold, -1, 1)

true_negatives = np.sum(preds_benign == 1)
false_positives = np.sum(preds_benign == -1)
true_positives = np.sum(preds_malicious == -1)
false_negatives = np.sum(preds_malicious == 1)

print("\n" + "="*40)
print("MAHALANOBIS ANOMALY DETECTION RESULTS")
print("="*40)
print(f"Benign Correctly Ignored (Normal):   {true_negatives} / {len(X_benign)}")
print(f"False Alarms (Benign as Anomaly):    {false_positives} / {len(X_benign)}")
print(f"Attacks Detected (True Anomaly):     {true_positives} / {len(X_malicious)}")
print(f"Missed Attacks (False Negative):     {false_negatives} / {len(X_malicious)}")
print("="*40)

# 5. Export to CSV
all_prompts = np.concatenate([prompts_benign, prompts_malicious])
all_ground_truth = ["Benign"] * len(X_benign) + ["Malicious"] * len(X_malicious)
all_labels = [1] * len(X_benign) + [-1] * len(X_malicious)
all_preds = np.concatenate([preds_benign, preds_malicious])
all_distances = np.concatenate([dist_benign, dist_malicious])

df = pd.DataFrame({
    "Ground_Truth": all_ground_truth,
    "Numeric_Label": all_labels,
    "Mahalanobis_Prediction": all_preds,
    "Distance_Score": all_distances,  # Higher distance = More anomalous
    "Prompt_Text": all_prompts
})

def evaluate_status(row):
    if row["Numeric_Label"] == -1 and row["Mahalanobis_Prediction"] == -1: return "True Anomaly"
    elif row["Numeric_Label"] == -1 and row["Mahalanobis_Prediction"] == 1: return "Missed Attack"
    elif row["Numeric_Label"] == 1 and row["Mahalanobis_Prediction"] == -1: return "False Alarm"
    else: return "Benign (Normal)"

df["Result_Status"] = df.apply(evaluate_status, axis=1)
df = df.sort_values(by="Distance_Score", ascending=False) # Highest distance at top

ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"./data/mahalanobis_results_{ts}.csv"
df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
print(f"\n✅ Exported detailed results to {csv_path}")