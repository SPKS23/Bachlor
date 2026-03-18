"""
Machine Learning Tree-Based Models:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Isolation Forest (Anomaly Detection)

Applied to Heart Disease Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# ── 1. Load Heart Disease Dataset ───────────────────────────────────
# Using the UCI Heart Disease dataset
from urllib.request import urlopen
import io

try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    df = pd.read_csv(url, header=None)
    
    # Column names for the heart disease dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df.columns = column_names
    
    # Replace '?' with NaN and drop rows with missing values
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    # Convert to numeric
    df = df.astype(float)
    
    # Binary classification: disease (1) vs no disease (0)
    df['target'] = (df['target'] > 0).astype(int)
    
except Exception as e:
    print(f"Error loading from URL: {e}")
    print("Creating synthetic heart disease data instead...")
    
    # Create synthetic heart disease data if URL fails
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=303,
        n_features=13,
        n_informative=8,
        n_redundant=3,
        n_classes=2,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Feature names
feature_names = X.columns.tolist()

print(f"Heart Disease Dataset loaded")
print(f"Dataset shape: {X.shape}")
print(f"Feature names: {feature_names}")
print(f"Target distribution:\n{y.value_counts()}\n")

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y,
)

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
    DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    X_train, X_test, y_train, y_test,
)

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    ax=ax,
    fontsize=9,
)
ax.set_title("Decision Tree (max_depth=5) – Heart Disease Classification", fontsize=14)
plt.tight_layout()
plt.savefig("decision_tree_heart.png", dpi=150)
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

# Feature importance bar chart
importances = rf_model.feature_importances_
idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(importances)), importances[idx], color="steelblue")
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
ax.set_title("Random Forest – Feature Importances (Heart Disease)")
ax.set_ylabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importances_heart.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════
# 4. GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════════
gb_model, gb_pred = evaluate(
    "Gradient Boosting",
    GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=RANDOM_STATE,
    ),
    X_train, X_test, y_train, y_test,
)

# ── Model comparison ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

models = {
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
}
accuracies = []
model_names = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    model_names.append(name)

ax.bar(model_names, accuracies, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison (Heart Disease)")
ax.set_ylim([0, 1])
for i, v in enumerate(accuracies):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison_heart.png", dpi=150)
plt.show()

# ── Confusion matrices side-by-side ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, name, pred in zip(
    axes,
    ["Decision Tree", "Random Forest", "Gradient Boosting"],
    [dt_pred, rf_pred, gb_pred],
):
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices_heart.png", dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════════════
# 5. ISOLATION FOREST  (Anomaly / Outlier Detection)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(" Isolation Forest – Anomaly Detection")
print(f"{'=' * 60}")

iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.1,   # expected fraction of outliers/anomalies
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
iso_pred = iso_model.fit_predict(X_test)   # 1 = inlier, -1 = outlier
scores = iso_model.decision_function(X_test)

n_outliers = (iso_pred == -1).sum()
print(f"  Detected anomalies : {n_outliers} / {len(X_test)}")
print(f"  Anomaly score range: [{scores.min():.3f}, {scores.max():.3f}]")

# Distribution of anomaly scores
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of anomaly scores
axes[0].hist(scores, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
axes[0].axvline(scores[iso_pred == -1].mean(), color="red", linestyle="--", linewidth=2, label="Anomaly Mean")
axes[0].set_xlabel("Anomaly Score")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Anomaly Scores")
axes[0].legend()

# Right: scatter plot of anomalies vs features
colors = np.where(iso_pred == 1, "steelblue", "red")
axes[1].scatter(range(len(scores)), scores, c=colors, s=50, alpha=0.6)
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Anomaly Score")
axes[1].set_title("Isolation Forest – Anomaly Scores\n(red = anomaly, blue = normal)")
axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("isolation_forest_heart.png", dpi=150)
plt.show()

# Display characteristics of detected anomalies
print("\n" + "=" * 60)
print(" Top Anomalous Cases (sorted by anomaly score)")
print("=" * 60)
anomaly_indices = np.where(iso_pred == -1)[0]
if len(anomaly_indices) > 0:
    top_anomalies = anomaly_indices[np.argsort(scores[anomaly_indices])[:5]]
    for i, idx in enumerate(top_anomalies, 1):
        print(f"\nAnomaly {i} (Index {idx}, Score {scores[idx]:.3f}):")
        print(f"  Actual: {'Disease' if y_test.iloc[idx] else 'No Disease'}")
        for j, feat in enumerate(feature_names):
            print(f"  {feat}: {X_test[idx, j]:.3f}")
else:
    print("No anomalies detected in the test set.")

print("\n✅ All models trained. Plots saved to the working directory.")
