import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

H = 0.03
RNG = np.random.RandomState(7)

# PU classification setting parameters
n = 3000  # null training sample size
k = 2500  # positive labeled samples
m = 1000  # test sample size
m0 = 500  # nulls in test sample
m1 = 500  # novelties in test sample
N_TOTAL = n + m  # total sample size


class MLPBinaryModule(nn.Module):
    def __init__(self, input_dim, hidden_dims=(20, 20)):
        super(MLPBinaryModule, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_dims[1], 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x.squeeze(1)


class TorchNNClassifier:
    def __init__(
        self,
        input_dim,
        hidden_dims=(20, 20),
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        n_epochs=120,
        random_state=2,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        Xs = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(Xs, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = MLPBinaryModule(self.input_dim, self.hidden_dims).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        X_tensor = torch.tensor(Xs, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            p1 = torch.sigmoid(logits).cpu().numpy()

        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def normal_pdf(x, mean, std=1.0):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))


def posterior_one_sided(x, pi0, pi1):
    p0 = normal_pdf(x, 0.0, 1.0)
    p1 = normal_pdf(x, 2.0, 1.0)
    return (pi1 * p1) / (pi0 * p0 + pi1 * p1)


def posterior_two_sided(x, pi0, pi1):
    p0 = normal_pdf(x, 0.0, 1.0)
    p1 = 0.5 * normal_pdf(x, 2.0, 1.0) + 0.5 * normal_pdf(x, -2.0, 1.0)
    return (pi1 * p1) / (pi0 * p0 + pi1 * p1)


def sample_1d_pu(two_sided=False):
    """Generate PU classification data for 1D setting."""
    # Y: null training sample (n samples from P0)
    Y = RNG.normal(loc=0.0, scale=1.0, size=n)
    
    # X: test sample (m0 from P0, m1 from P1)
    X_0 = RNG.normal(loc=0.0, scale=1.0, size=m0)
    if two_sided:
        signs = RNG.choice([-1.0, 1.0], size=m1)
        X_1 = RNG.normal(loc=2.0 * signs, scale=1.0, size=m1)
    else:
        X_1 = RNG.normal(loc=2.0, scale=1.0, size=m1)
    
    # Z: full sample = [Y, X]
    Z = np.concatenate([Y, X_0, X_1])[:, None]
    
    # True labels: [0]*n + [0]*m0 + [1]*m1
    y_true = np.concatenate([np.zeros(n, dtype=int), np.zeros(m0, dtype=int), np.ones(m1, dtype=int)])
    
    # PU labels: [-1]*k + [1]*(n+m-k)
    y_pu = np.concatenate([np.full(k, -1, dtype=int), np.ones(n + m - k, dtype=int)])
    
    return Z, y_true, y_pu


def sample_2d_pu(two_sided=False):
    """Generate PU classification data for 2D setting."""
    # Y: null training sample (n samples from P0)
    Y = RNG.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(n, 2))
    
    # X: test sample (m0 from P0, m1 from P1)
    X_0 = RNG.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(m0, 2))
    if two_sided:
        mix = RNG.binomial(1, 0.5, size=m1)
        means = np.where(mix[:, None] == 1, [2.0, 2.0], [-2.0, -2.0])
        X_1 = RNG.normal(loc=means, scale=[1.0, 1.0], size=(m1, 2))
    else:
        X_1 = RNG.normal(loc=[2.0, 2.0], scale=[1.0, 1.0], size=(m1, 2))
    
    # Z: full sample = [Y, X]
    Z = np.vstack([Y, X_0, X_1])
    
    # True labels: [0]*n + [0]*m0 + [1]*m1
    y_true = np.concatenate([np.zeros(n, dtype=int), np.zeros(m0, dtype=int), np.ones(m1, dtype=int)])
    
    # PU labels: [-1]*k + [1]*(n+m-k)
    y_pu = np.concatenate([np.full(k, -1, dtype=int), np.ones(n + m - k, dtype=int)])
    
    return Z, y_true, y_pu


def build_models(input_dim):
    return [
        (
            "Logistic Regression",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
        ),
        (
            "Neural Network",
            TorchNNClassifier(
                input_dim=input_dim,
                hidden_dims=(20, 20),
                lr=1e-3,
                weight_decay=2e-4,
                batch_size=256,
                n_epochs=120,
                random_state=2,
            ),
        ),
    ]


fig, axes = plt.subplots(4, 4, figsize=(13.8, 10.8))
cm_bright = ListedColormap(["#ff7f0e", "#6aa0ff"])
cm = plt.cm.RdBu

# ------------------------
# Top block: d = 1 settings
# ------------------------
for row, two_sided in enumerate([False, True]):
    Z, y_true, y_pu = sample_1d_pu(two_sided=two_sided)
    
    # Compute posterior probabilities for true labels
    pi0 = m0 / m  # proportion of nulls in test sample
    pi1 = m1 / m  # proportion of novelties in test sample
    x_grid = np.linspace(-4, 4, 600)[:, None]
    if two_sided:
        g_star = posterior_two_sided(x_grid[:, 0], pi0, pi1)
    else:
        g_star = posterior_one_sided(x_grid[:, 0], pi0, pi1)

    # True labels histogram (based on true class membership)
    ax = axes[row, 0]
    ax.hist(Z[y_true == 0, 0], bins=24, alpha=0.9, color="#1f77b4", label=r"$P_0$")
    ax.hist(Z[y_true == 1, 0], bins=24, alpha=0.7, color="#ff7f0e", label=r"$P_1$")
    ax.set_xlim(-4, 4)
    ax.legend(loc="upper left", fontsize=8)

    # Classification labels histogram (PU labels)
    ax = axes[row, 1]
    ax.hist(Z[y_pu == -1, 0], bins=24, alpha=0.9, color="#1f77b4", label="Labeled")
    ax.hist(Z[y_pu == 1, 0], bins=24, alpha=0.7, color="#ff7f0e", label="Unlabeled")
    ax.set_xlim(-4, 4)
    ax.legend(loc="upper left", fontsize=8)

    # Convert PU labels to binary: -1 (labeled) -> 1, 1 (unlabeled) -> 0
    y_binary = (y_pu == -1).astype(int)
    
    # Train on binary labels
    X_train, X_test, y_train, y_test = train_test_split(
        Z, y_binary, test_size=0.35, random_state=42, stratify=y_binary
    )

    row_models = build_models(input_dim=1)

    for col_offset, (name, clf) in enumerate(row_models, start=2):
        ax = axes[row, col_offset]
        clf.fit(X_train, y_train)
        # Plot probability of being unlabeled/novel (class 0)
        g_hat = 1 - clf.predict_proba(x_grid)[:, 1]
        ax.plot(x_grid[:, 0], g_hat, color="#5DA5DA", linewidth=1.3, label=r"$\hat{g}$")
        ax.plot(x_grid[:, 0], g_star, color="#F17CB0", linewidth=1.0, label=r"$g^*$")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=8)


# ---------------------------
# Bottom block: d = 2 settings
# ---------------------------
for row_offset, two_sided in enumerate([False, True], start=2):
    Z, y_true, y_pu = sample_2d_pu(two_sided=two_sided)

    x_min, x_max = Z[:, 0].min() - 0.6, Z[:, 0].max() + 0.6
    y_min, y_max = Z[:, 1].min() - 0.6, Z[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.arange(x_min, x_max, H), np.arange(y_min, y_max, H))

    # True labels scatter
    ax = axes[row_offset, 0]
    ax.scatter(Z[:, 0], Z[:, 1], c=y_true, cmap=cm_bright, s=5, edgecolors="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    # Classification labels scatter (PU labels)
    ax = axes[row_offset, 1]
    ax.scatter(Z[:, 0], Z[:, 1], c=(y_pu == -1).astype(int), cmap=cm_bright, s=5, edgecolors="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    # Convert PU labels to binary: -1 (labeled) -> 1, 1 (unlabeled) -> 0
    y_binary = (y_pu == -1).astype(int)
    
    # Track indices to map back to true labels
    indices = np.arange(len(Z))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        Z, y_binary, indices, test_size=0.35, random_state=42, stratify=y_binary
    )
    
    # Get true labels for train/test samples
    y_true_train = y_true[idx_train]
    y_true_test = y_true[idx_test]

    row_models = build_models(input_dim=2)

    for col_offset, (name, clf) in enumerate(row_models, start=2):
        ax = axes[row_offset, col_offset]
        clf.fit(X_train, y_train)

        if hasattr(clf, "decision_function"):
            Z_pred = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
        else:
            Z_pred = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]
        Z_pred = Z_pred.reshape(xx.shape)
        # Show probability of class 0 (unlabeled/novel)
        Z_pred = 1 - Z_pred

        contour = ax.contourf(xx, yy, Z_pred, cmap=cm, alpha=0.8, levels=12)
        # Plot points colored by TRUE labels, not PU labels
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_true_train, cmap=cm_bright, s=5, edgecolors="none")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_true_test, cmap=cm_bright, alpha=0.6, s=5, edgecolors="none")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        fig.colorbar(contour, ax=ax, fraction=0.047, pad=0.01)


titles = ["True labels", "Classification labels", "Logistic Regression", "Neural Network"]
for col, title in enumerate(titles):
    axes[0, col].set_title(title, fontsize=15)

for row in range(4):
    for col in range(4):
        for spine in axes[row, col].spines.values():
            spine.set_alpha(0.7)

caption = (
    "Fig. 4.  Plot of $g^*$ and $\\hat{g}$ in different settings (rows) with different loss functions "
    "$\\ell(\\cdot,\\cdot)$ and function classes $\\mathcal{G}$ (with default parameters in scikit-learn). "
    "In all settings, $m=1000$, $m_0=500$, $m_1=500$, $n=3000$, and $k=2000$. "
    "The top two rows correspond to $d=1$ and the bottom two rows correspond to $d=2$. "
    "For the first and third rows, $P_1=\\mathcal{N}((2,\\ldots,2), I_d)$ (one-sided alternatives); "
    "for the second and fourth rows, $P_1=0.5\\,\\mathcal{N}((2,\\ldots,2), I_d)+0.5\\,\\mathcal{N}((-2,\\ldots,-2), I_d)$ "
    "(two-sided alternatives)."
)

fig.text(0.05, 0.02, caption, fontsize=10, ha="left", va="bottom", family="serif", style="italic")
fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12, wspace=0.18, hspace=0.25)

plt.savefig("Figur42_recreated.png", dpi=200, bbox_inches="tight")
plt.show()