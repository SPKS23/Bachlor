"""
Machine Learning Tree-Based Models:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Isolation Forest (Anomaly Detection)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
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

# ── 1. Generate synthetic classification data ────────────────────────
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=RANDOM_STATE,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("Dataset shape:", df.shape)
print(df.head(), "\n")

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

fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(
    dt_model,
    feature_names=feature_names,
    class_names=["Class 0", "Class 1"],
    filled=True,
    rounded=True,
    ax=ax,
    fontsize=8,
)
ax.set_title("Decision Tree (max_depth=5)", fontsize=14)
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

# Feature importance bar chart
importances = rf_model.feature_importances_
idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(importances)), importances[idx], color="steelblue")
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
ax.set_title("Random Forest – Feature Importances")
ax.set_ylabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importances.png", dpi=150)
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

# ── ROC curves for the three classifiers ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

for name, model in [
    ("Decision Tree", dt_model),
    ("Random Forest", rf_model),
    ("Gradient Boosting", gb_model),
]:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves – Classifier Comparison")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
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
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════════════
# 5. ISOLATION FOREST  (Anomaly / Outlier Detection)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(" Isolation Forest – Anomaly Detection")
print(f"{'=' * 60}")

# Create data with a small cluster of outliers
X_normal, _ = make_blobs(
    n_samples=800, centers=1, cluster_std=1.0, random_state=RANDOM_STATE,
)
X_outliers = np.random.uniform(low=-8, high=8, size=(50, 2))
X_iso = np.vstack([X_normal, X_outliers])

iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.06,   # expected fraction of outliers
    random_state=RANDOM_STATE,
)
iso_pred = iso_model.fit_predict(X_iso)   # 1 = inlier, -1 = outlier
scores = iso_model.decision_function(X_iso)

n_outliers = (iso_pred == -1).sum()
print(f"  Detected outliers : {n_outliers} / {len(X_iso)}")
print(f"  Anomaly score range: [{scores.min():.3f}, {scores.max():.3f}]")

# Scatter plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: predictions
colors = np.where(iso_pred == 1, "steelblue", "red")
axes[0].scatter(X_iso[:, 0], X_iso[:, 1], c=colors, s=15, alpha=0.7)
axes[0].set_title("Isolation Forest – Predictions\n(red = outlier)")
axes[0].set_xlabel("Feature 0")
axes[0].set_ylabel("Feature 1")

# Right: anomaly score heat-map
sc = axes[1].scatter(
    X_iso[:, 0], X_iso[:, 1], c=scores, cmap="coolwarm", s=15, alpha=0.7,
)
axes[1].set_title("Isolation Forest – Anomaly Scores")
axes[1].set_xlabel("Feature 0")
axes[1].set_ylabel("Feature 1")
plt.colorbar(sc, ax=axes[1], label="Anomaly Score")

plt.tight_layout()
plt.savefig("isolation_forest.png", dpi=150)
plt.show()

print("\n✅ All models trained. Plots saved to the working directory.")
