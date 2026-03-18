"""
Machine Learning Tree-Based Models:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Isolation Forest (Anomaly Detection)
  - Neural Network with AdaDetect (False Discovery Rate Control)

Applied to MNIST Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)



# ── 0. Reproducibility ──────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── 0.5 Helper: FDP/TDP Statistics ──────────────────────────────────
def get_fdp_tdp(ytrue, scores_or_preds, threshold=None):
    """Compute False Discovery Proportion and True Discovery Proportion."""
    if threshold is None:
        predictions = (scores_or_preds > np.median(scores_or_preds)).astype(int)
    else:
        predictions = (scores_or_preds > threshold).astype(int)
    
    rejection_set = np.where(predictions == 1)[0]
    if len(rejection_set) > 0:
        fdp = np.sum((ytrue[rejection_set] == 0)) / len(rejection_set)
        tdp = np.sum((ytrue[rejection_set] == 1)) / (np.sum(ytrue == 1) + 1e-10)
    else: 
        fdp = 0
        tdp = 0
    return fdp, tdp

# ── 1. Load MNIST data from IDX format files ────────────────────────
def load_mnist_images(filename):
    """Load MNIST images from IDX format."""
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.fromfile(f, dtype=np.uint8, count=num * rows * cols)
        return data.reshape(num, rows * cols)

def load_mnist_labels(filename):
    """Load MNIST labels from IDX format."""
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.uint8, count=num)
        return data

# Load MNIST data
data_path = "Soren/Adadetect/data/MNIST/raw/"
X_train_full = load_mnist_images(data_path + "train-images-idx3-ubyte")
y_train_full = load_mnist_labels(data_path + "train-labels-idx1-ubyte")
X_test_full = load_mnist_images(data_path + "t10k-images-idx3-ubyte")
y_test_full = load_mnist_labels(data_path + "t10k-labels-idx1-ubyte")

# Normalize pixel values to [0, 1]
X_train_full = X_train_full.astype(np.float32) / 255.0
X_test_full = X_test_full.astype(np.float32) / 255.0

# Use a subset for faster training (10k training samples, 2k test samples)
n_train = 10000
n_test = 2000
indices_train = np.random.choice(len(X_train_full), n_train, replace=False)
indices_test = np.random.choice(len(X_test_full), n_test, replace=False)

X_train_subset = X_train_full[indices_train]
y_train_subset = y_train_full[indices_train]
X_test_subset = X_test_full[indices_test]
y_test_subset = y_test_full[indices_test]

# Final train/test split
X_train = X_train_subset
y_train = y_train_subset
X_test = X_test_subset
y_test = y_test_subset

print(f"MNIST Dataset loaded")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")
print(f"Class distribution: {np.bincount(y_train)}\n")

# ── helper: evaluate & print ─────────────────────────────────────────
def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    """Fit, predict, and print metrics."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="accuracy")
    print(f"\n{'=' * 60}")
    print(f" {name}")
    print(f"{'=' * 60}")
    print(f"  Test accuracy : {acc:.4f}")
    print(f"  CV accuracy   : {cv.mean():.4f} ± {cv.std():.4f}")
    print(classification_report(y_te, y_pred, zero_division=0))
    return model, y_pred


# ══════════════════════════════════════════════════════════════════════
# 2. DECISION TREE
# ══════════════════════════════════════════════════════════════════════
dt_model, dt_pred = evaluate(
    "Decision Tree",
    DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    X_train, X_test, y_train, y_test,
)

fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(
    dt_model,
    class_names=[str(i) for i in range(10)],
    filled=True,
    rounded=True,
    ax=ax,
    fontsize=8,
)
ax.set_title("Decision Tree (max_depth=8) – MNIST Digits Classification", fontsize=14)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════
# 3. RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════
rf_model, rf_pred = evaluate(
    "Random Forest",
    RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    X_train, X_test, y_train, y_test,
)

# Feature importance bar chart (top 30 features for MNIST)
importances = rf_model.feature_importances_
idx = np.argsort(importances)[::-1][:30]  # Top 30 features

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(idx)), importances[idx], color="steelblue")
ax.set_xticks(range(len(idx)))
ax.set_xticklabels([f"pixel_{i}" for i in idx], rotation=45, ha="right", fontsize=8)
ax.set_title("Random Forest – Top 30 Feature Importances (MNIST)")
ax.set_ylabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importances.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════
# 4. GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════════
# gb_model, gb_pred = evaluate(
#     "Gradient Boosting",
#     GradientBoostingClassifier(
#         n_estimators=50,  # Further reduced for speed
#         learning_rate=0.2,  # Higher learning rate = faster convergence
#         max_depth=2,  # Shallow trees train much faster
#         subsample=0.7,  # More aggressive sampling
#         validation_fraction=0.1,  # Use early stopping
#         n_iter_no_change=5,  # Stop if no improvement for 5 iterations
#         random_state=RANDOM_STATE,
#     ),
#     X_train, X_test, y_train, y_test,
# )

# # ── Model accuracy comparison ────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(8, 6))

# models = {
#     "Decision Tree": dt_model,
#     "Random Forest": rf_model,
#     "Gradient Boosting": gb_model,
# }
# accuracies = []
# model_names = []

# for name, model in models.items():
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     accuracies.append(acc)
#     model_names.append(name)

# ax.bar(model_names, accuracies, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
# ax.set_ylabel("Accuracy")
# ax.set_title("Model Accuracy Comparison (MNIST)")
# ax.set_ylim([0, 1])
# for i, v in enumerate(accuracies):
#     ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
# plt.tight_layout()
# plt.savefig("model_comparison.png", dpi=150)
# plt.show()

# # ── Confusion matrices side-by-side ──────────────────────────────────
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# for ax, name, pred in zip(
#     axes,
#     ["Decision Tree", "Random Forest", "Gradient Boosting"],
#     [dt_pred, rf_pred, gb_pred],
# ):
#     ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues")
#     ax.set_title(name)
# plt.tight_layout()
# plt.savefig("confusion_matrices.png", dpi=150)
# plt.show()


# ══════════════════════════════════════════════════════════════════════
# 5. ISOLATION FOREST  (Anomaly / Outlier Detection on MNIST)
# ══════════════════════════════════════════════════════════════════════
# ── CONFIGURATION: Change these to test different digit combinations ──
INLIER_DIGIT = 5        # Which digit is "normal/common"
OUTLIER_DIGIT = 4       # Which digit is "anomaly/rare"
# ──────────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f" Isolation Forest – Digit {INLIER_DIGIT} (common) vs Digit {OUTLIER_DIGIT} (anomaly)")
print(f"{'=' * 60}")

# Create dataset with mostly inlier digit for training
inlier_mask = y_train == INLIER_DIGIT
outlier_mask = y_train == OUTLIER_DIGIT

X_inlier = X_train[inlier_mask]

# Train on inlier samples (learn what "normal" looks like)
iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.1,   # expect 10% outliers in test
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
iso_model.fit(X_inlier)

# Create test set with mix of inliers and outliers
X_test_inlier = X_train[inlier_mask][:500]  # First 500 inliers
X_test_outlier = X_train[outlier_mask][:500] if outlier_mask.sum() >= 500 else X_train[outlier_mask]

X_test_iso = np.vstack([X_test_inlier, X_test_outlier])
y_test_iso = np.hstack([np.zeros(len(X_test_inlier)), np.ones(len(X_test_outlier))])

# Predictions and scores
iso_pred = iso_model.predict(X_test_iso)   # 1 = inlier, -1 = outlier
scores = iso_model.decision_function(X_test_iso)

n_outliers_detected = (iso_pred == -1).sum()
print(f"  Test set: {len(X_test_inlier)} digit {INLIER_DIGIT}s, {len(X_test_outlier)} digit {OUTLIER_DIGIT}s")
print(f"  Detected as anomalies: {n_outliers_detected} / {len(X_test_iso)}")
print(f"  Anomaly score range: [{scores.min():.3f}, {scores.max():.3f}]")

# Visualize some anomalous vs normal digits
anomaly_indices = np.where(iso_pred == -1)[0]
normal_indices = np.where(iso_pred == 1)[0]

# Get indices of most anomalous and most normal samples
most_anomalous = anomaly_indices[np.argsort(scores[anomaly_indices])[:8]]
most_normal = normal_indices[np.argsort(-scores[normal_indices])[:8]]

fig, axes = plt.subplots(2, 8, figsize=(14, 4))

# Top row: Most anomalous
for i, idx in enumerate(most_anomalous):
    axes[0, i].imshow(X_test_iso[idx].reshape(28, 28), cmap="gray")
    true_digit = OUTLIER_DIGIT if y_test_iso[idx] == 1 else INLIER_DIGIT
    axes[0, i].set_title(f"Digit: {true_digit}\nScore: {scores[idx]:.2f}", fontsize=8)
    axes[0, i].axis("off")

# Bottom row: Most normal
for i, idx in enumerate(most_normal):
    axes[1, i].imshow(X_test_iso[idx].reshape(28, 28), cmap="gray")
    true_digit = OUTLIER_DIGIT if y_test_iso[idx] == 1 else INLIER_DIGIT
    axes[1, i].set_title(f"Digit: {true_digit}\nScore: {scores[idx]:.2f}", fontsize=8)
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Most Anomalous", fontweight="bold")
axes[1, 0].set_ylabel("Most Normal", fontweight="bold")
plt.suptitle(f"Isolation Forest – Digit {INLIER_DIGIT} (common) vs {OUTLIER_DIGIT} (anomaly)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("isolation_forest.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════
# 6. TREE-BASED ANOMALY DETECTION WITH FDP/TDP (digit 4 vs 7)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(" FDP/TDP Evaluation: All Methods on Digit 4 (Inlier) vs 7 (Outlier)")
print(f"{'=' * 60}")

# Extract samples where label is 4 or 7
mask_4 = y_train == 4
mask_7 = y_train == 7
mask_combined = mask_4 | mask_7

X_anomaly = X_train[mask_combined]
y_anomaly_raw = y_train[mask_combined]

# Remap labels: 4 -> 0 (inlier), 7 -> 1 (outlier)
y_anomaly = (y_anomaly_raw == 7).astype(int)

# Use 60% for training, 40% for testing
idx_split = int(0.6 * len(X_anomaly))
X_anom_train = X_anomaly[:idx_split]
y_anom_train = y_anomaly[:idx_split]
X_anom_test = X_anomaly[idx_split:]
y_anom_test = y_anomaly[idx_split:]

print(f"\n  Training set: {len(X_anom_train)} samples ({np.sum(y_anom_train == 0)} inliers, {np.sum(y_anom_train == 1)} outliers)")
print(f"  Test set: {len(X_anom_test)} samples ({np.sum(y_anom_test == 0)} inliers, {np.sum(y_anom_test == 1)} outliers)\n")

# Evaluate each model
models_eval = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, max_depth=2, random_state=RANDOM_STATE),
    "Isolation Forest": IsolationForest(n_estimators=200, contamination=0.1, random_state=RANDOM_STATE, n_jobs=-1),
}

results_fdp_tdp = {}

for name, model in models_eval.items():
    print(f"  {name}...", end=" ", flush=True)
    
    if "Isolation" in name:
        # Isolation Forest doesn't need training labels, use anomaly scores directly
        model.fit(X_anom_train)
        scores = model.decision_function(X_anom_test)
    else:
        # Train on labeled data (4 vs 7 binary classification)
        model.fit(X_anom_train, y_anom_train)
        # Get predictions (0 or 1)
        preds = model.predict(X_anom_test)
        scores = preds
    
    # Compute FDP/TDP
    fdp, tdp = get_fdp_tdp(y_anom_test, scores)
    results_fdp_tdp[name] = (fdp, tdp)
    print(f"FDP={fdp:.3f}, TDP={tdp:.3f}")

# Summary
print(f"\n{'─' * 60}")
print(f"  Summary:")
print(f"{'─' * 60}")
for name, (fdp, tdp) in sorted(results_fdp_tdp.items(), key=lambda x: -x[1][1]):  # Sort by TDP descending
    print(f"  {name:20s} | FDP={fdp:.3f} | TDP={tdp:.3f}")

print("\n✅ All models trained. Plots saved to the working directory.")
