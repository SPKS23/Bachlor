import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# =========================
# External BH procedures
# =========================
try:
    from .algo import EmpBH, adaptiveEmpBH
except ImportError:
    from algo import EmpBH, adaptiveEmpBH


# =========================
# FDP / TDP
# =========================
def get_fdp(ytrue, rejection_set):
    if len(rejection_set):
        fdp = np.sum(ytrue[rejection_set] == 0) / len(rejection_set)
        tdp = np.sum(ytrue[rejection_set] == 1) / np.sum(ytrue == 1)
    else:
        fdp, tdp = 0, 0
    return fdp, tdp


# =========================
# Logistic Regression Model
# =========================
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.linear(x)


class LogisticClassifier:
    def __init__(self, input_dim, batch_size=128, n_epochs=10):
        self.model = LogisticRegressionModel(input_dim)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.n_epochs):
            for x, y in loader:
                self.model.train()
                y = y.unsqueeze(-1).float()

                logits = self.model(x)
                loss = self.loss_fn(logits, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def predict_proba(self, dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        with torch.no_grad():
            for x, _ in loader:
                self.model.eval()
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                return probs.numpy().flatten()


# =========================
# Custom dataset wrapper
# =========================
class custom_subset(Dataset):
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = torch.tensor(labels)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return image, target

    def __len__(self):
        return len(self.targets)


# =========================
# Load MNIST
# =========================
train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

# Define null (4) and alternative (9)
inlr_labels = [4]
outlr_labels = [9]

targets = torch.tensor(train_data.targets)

inlr = (targets[..., None] == torch.tensor(inlr_labels)).any(-1).nonzero(as_tuple=True)[0]
outlr = (targets[..., None] == torch.tensor(outlr_labels)).any(-1).nonzero(as_tuple=True)[0]

print("Nulls:", len(inlr), "Outliers:", len(outlr))


# =========================
# Experiment parameters
# =========================
level = 0.1
test_size = 1000
null_prop = 0.9

n_inlr = int(test_size * null_prop)
n_outlr = test_size - n_inlr

calib_size = 1000
train_size = 3000
n_runs = 10

fdp, tdp = [0.] * n_runs, [0.] * n_runs


# =========================
# Main loop
# =========================
for i in range(n_runs):

    # ---- Test set ----
    test_idx = np.concatenate([
        np.random.choice(inlr, n_inlr, replace=False),
        np.random.choice(outlr, n_outlr, replace=False)
    ])

    test_labels = np.concatenate([np.zeros(n_inlr), np.ones(n_outlr)])
    test_dataset = custom_subset(train_data, test_idx, test_labels)

    # ---- Training + calibration ----
    train_idx = np.setdiff1d(inlr.numpy(), test_idx)
    np.random.shuffle(train_idx)

    calib_idx = train_idx[:calib_size]
    train_idx2 = train_idx[calib_size:calib_size + train_size]

    calib_dataset = custom_subset(train_data, calib_idx, np.zeros(calib_size))
    train_dataset = custom_subset(train_data, train_idx2, np.zeros(train_size))

    # Combine all (same as your original logic)
    training_dataset = ConcatDataset([train_dataset, calib_dataset, test_dataset])

    # ---- Train logistic regression ----
    clf = LogisticClassifier(input_dim=28 * 28, batch_size=32, n_epochs=10)
    clf.fit(training_dataset)

    # ---- Scores ----
    test_scores = clf.predict_proba(test_dataset)
    null_scores = clf.predict_proba(calib_dataset)

    rej_set = EmpBH(null_scores, test_scores, level=level)

    fdp_, tdp_ = get_fdp(test_labels, rej_set)

    fdp[i] = fdp_
    tdp[i] = tdp_


# =========================
# Results
# =========================
print("FDP:", np.mean(fdp), np.std(fdp))
print("TDP:", np.mean(tdp), np.std(tdp))


# =========================
# Visualization
# =========================
idx_rej = rej_set
idx_acc = np.setdiff1d(np.arange(len(test_dataset)), idx_rej)

sample_rej = np.random.choice(idx_rej, size=10, replace=False)
sample_acc = np.random.choice(idx_acc, size=38, replace=False)

samples = np.concatenate([sample_rej, sample_acc])
np.random.shuffle(samples)

images = [test_dataset[i][0][0] for i in samples]

fig = plt.figure(figsize=(6, 6))
grid = ImageGrid(fig, 111, nrows_ncols=(6, 4), axes_pad=0.05)

for ax, idx, im in zip(grid, samples, images):
    ax.imshow(im, cmap="gray")
    ax.axis("off")

    if idx in sample_rej:
        for spine in ax.spines.values():
            spine.set_color("crimson")
            spine.set_linewidth(3)

plt.show()