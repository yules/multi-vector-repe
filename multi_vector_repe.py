import numpy as np
import pandas as pd
import csv
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans

# 1. Load cached vectors
print("Loading cached vectors...")
data = np.load("./data/latent_vectors.npz", allow_pickle=True)
X_benign = data["X_benign"]
X_malicious = data["X_malicious"]
prompts_benign = data["prompts_benign"]
prompts_malicious = data["prompts_malicious"]

# 2. Train/Test Split (100 Calibration / Rest for Testing)
np.random.seed(42)
calib_benign_idx = np.random.choice(len(X_benign), size=50, replace=False)
calib_malicious_idx = np.random.choice(len(X_malicious), size=50, replace=False)

test_benign_idx = np.setdiff1d(np.arange(len(X_benign)), calib_benign_idx)
test_malicious_idx = np.setdiff1d(np.arange(len(X_malicious)), calib_malicious_idx)

X_calib_benign = X_benign[calib_benign_idx]
X_calib_malicious = X_malicious[calib_malicious_idx]

# 3. Multi-Vector Extraction (K-Means on Malicious Intents)
K = 4
print(f"Clustering malicious calibration set into {K} distinct sub-manifolds...")
kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
kmeans.fit(X_calib_malicious)

# The benign baseline remains a single dense manifold
mu_benign = np.mean(X_calib_benign, axis=0)

# Create K orthogonal harm vectors
harm_vectors = []
for i in range(K):
    centroid = kmeans.cluster_centers_[i]
    vector = centroid - mu_benign
    vector = vector / np.linalg.norm(vector)
    harm_vectors.append(vector)

# 4. Evaluate Held-Out Prompts
X_test = np.vstack([X_benign[test_benign_idx], X_malicious[test_malicious_idx]])
prompts_test = np.concatenate([prompts_benign[test_benign_idx], prompts_malicious[test_malicious_idx]])
ground_truth_test = ["Benign"] * len(test_benign_idx) + ["Malicious"] * len(test_malicious_idx)
y_true = np.array([0] * len(test_benign_idx) + [1] * len(test_malicious_idx))

# Score every test prompt against ALL K vectors
# Shape will be (N_test_samples, K)
all_scores = np.dot(X_test, np.array(harm_vectors).T)

# The final anomaly score is the max alignment with ANY of the harm sub-manifolds
max_scores = np.max(all_scores, axis=1)

# 5. Threshold & Evaluation
benign_scores = max_scores[:len(test_benign_idx)]
threshold = np.percentile(benign_scores, 99)
y_pred = (max_scores > threshold).astype(int)

auc = roc_auc_score(y_true, max_scores)
print("\n" + "=" * 40)
print(f"MULTI-VECTOR RepE RESULTS (K={K})")
print("=" * 40)
print(f"ROC-AUC Score: {auc:.4f}")
print(f"Decision Threshold (99th %ile Benign): {threshold:.4f}")
print(f"Benign Correctly Passed:  {np.sum(y_pred[:len(test_benign_idx)] == 0)} / {len(test_benign_idx)}")
print(f"False Alarms (FP):         {np.sum(y_pred[:len(test_benign_idx)] == 1)} / {len(test_benign_idx)}")
print(f"Attacks Caught (TP):       {np.sum(y_pred[len(test_benign_idx):] == 1)} / {len(test_malicious_idx)}")
print(f"Attacks Missed (FN):       {np.sum(y_pred[len(test_benign_idx):] == 0)} / {len(test_malicious_idx)}")
print("=" * 40)

# 6. Export Held-Out Results to CSV
df = pd.DataFrame({
    "Ground_Truth": ground_truth_test,
    "RepE_Harm_Score": max_scores,
    "Prediction": ["Malicious" if p == 1 else "Benign" for p in y_pred],
    "Prompt_Text": prompts_test
})

def evaluate_status(row):
    if row["Ground_Truth"] == "Malicious" and row["Prediction"] == "Malicious":
        return "True Anomaly (Detected Attack)"
    elif row["Ground_Truth"] == "Malicious" and row["Prediction"] == "Benign":
        return "Missed Attack (False Negative)"
    elif row["Ground_Truth"] == "Benign" and row["Prediction"] == "Malicious":
        return "False Alarm (False Positive)"
    return "Benign (Passed)"

df["Result_Status"] = df.apply(evaluate_status, axis=1)
df = df.sort_values(by="RepE_Harm_Score", ascending=False)

ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"./data/multi_vector_repe_results_{ts}.csv"
df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\")
print(f"\nSaved evaluation metrics and prompts to {csv_path}")