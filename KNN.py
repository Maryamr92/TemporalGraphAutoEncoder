from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import pickle
import re
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from preparing_data import load_split_factors
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from sklearn.metrics import roc_auc_score



factors_dir_normal="1sample-0.005-5000nodes/normalTest/"
factors_dir_abnormal="1sample-0.005-5000nodes/abnormalTest/"
factors_dir_training="1sample-0.005-5000nodes/"

training_set, _ = load_split_factors(factors_dir_training, train_ratio=1.0, val_ratio=0.0)
# print('training set is loaded')
test1, test_factors_normal = load_split_factors(factors_dir_normal, train_ratio=0.7, val_ratio=0.3)
# print('test_factors_normal set is loaded')
test2, test_factors_abnormal= load_split_factors(factors_dir_abnormal, train_ratio=0.7, val_ratio=0.3)
# print('test_factors_abnormal set is loaded')

# print(f'test_factors_normal, {test_factors_normal}')
from collections import Counter

# === Prepare training data (normal samples only) ===
A_train_vectors = [a.reshape(-1) for a in [f[0] for f in training_set]]
B_train_vectors = [b.reshape(-1) for b in [f[1] for f in training_set]]
C_train_vectors = [c.reshape(-1) for c in [f[2] for f in training_set]]

ABC_train_vectors = [
    np.concatenate([a, b, c])
    for a, b, c in zip(A_train_vectors, B_train_vectors, C_train_vectors)
]

# === Prepare test data (normal + abnormal1) ===
A_vectors = [a.reshape(-1) for a in [f[0] for f in test_factors_normal + test_factors_abnormal]]
B_vectors = [b.reshape(-1) for b in [f[1] for f in test_factors_normal + test_factors_abnormal]]
C_vectors = [c.reshape(-1) for c in [f[2] for f in test_factors_normal + test_factors_abnormal]]

ABC_vectors = [
    np.concatenate([a, b, c])
    for a, b, c in zip(A_vectors, B_vectors, C_vectors)
]

print(f'ABC_vectors.shape, {len(ABC_vectors)}, {ABC_vectors[0].shape}')
# === KNN Outlier Detection on combined ABC ===
print("KNN on combined A+B+C ---------------------------")
n_neighbors = 35

nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(ABC_train_vectors)
distances, indices = nbrs.kneighbors(ABC_vectors)

# Use mean distance to k nearest neighbors (excluding self if present)
outlier_scores = distances[:, 1:n_neighbors].mean(axis=1)


num_normal = len(test_factors_normal)  #
num_abnormal = len(test_factors_abnormal)
# Create labels: 0 for normal, 1 for abnormal1
labels = [0] * num_normal + [1] * num_abnormal


auc_score = roc_auc_score(labels, outlier_scores)
print(f"AUC-ROC: {auc_score:.4f}")


# Choose threshold (e.g., 90th percentile for top 10% as outliers)
threshold = np.percentile(outlier_scores, 5)
outliers = np.where(outlier_scores > threshold)[0]

print(f"Outlier indices (threshold = {threshold:.4f}):", outliers.tolist())
# Number of normal and abnormal1 samples
y_pred = [1 if score > threshold else 0 for score in outlier_scores]


precision = precision_score(labels, y_pred)
recall = recall_score(labels, y_pred)
f1 = f1_score(labels, y_pred)
tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()

print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
print(f"False Positives: {fp} / {len([l for l in labels if l == 0])}")
print(f"True Positives:  {tp} / {len([l for l in labels if l == 1])}")

############# IsolationForest ############

print('IsolationForest --------------------------------')
# === Isolation Forest ===
model = IsolationForest(contamination=0.5, random_state=42)
model.fit(ABC_train_vectors)

# Predict: -1 = outlier, 1 = normal
outlier_labels = model.predict(ABC_vectors)
outlier_indices = np.where(outlier_labels == -1)[0]

print(f"Detected {len(outlier_indices)} outliers at indices: {outlier_indices.tolist()}")

anomaly_scores = -model.decision_function(ABC_vectors)

# Calculate AUC-ROC using scores
auc_iso = roc_auc_score(labels, anomaly_scores)
print(f"Isolation Forest AUC-ROC: {auc_iso:.4f}")

# Predicted labels from model.predict (convert from {-1,1} to {1,0} for abnormal1, normal)
pred_labels = (outlier_labels == -1).astype(int)  # 1 = outlier, 0 = normal

# Calculate F1-score (and other metrics if you want)
f1 = f1_score(labels, pred_labels)
precision = precision_score(labels, pred_labels)
recall = recall_score(labels, pred_labels)

print(f"Isolation Forest F1-score: {f1:.4f}")
print(f"Isolation Forest Precision: {precision:.4f}")
print(f"Isolation Forest Recall: {recall:.4f}")








