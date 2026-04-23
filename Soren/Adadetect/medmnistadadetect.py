import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# MedMNIST
import medmnist
from medmnist import INFO

# =========================
# BH procedures (your import)
# =========================
try:
    from .algo import EmpBH, adaptiveEmpBH
except ImportError:
    from algo import EmpBH, adaptiveEmpBH


# =========================
# FDP / TDP
# =========================
def get_fdp(ytrue, rejection_set):
    if rejection_set.size:
        fdp = np.sum(ytrue[rejection_set] == 0) / len(rejection_set)
        tdp = np.sum(ytrue[rejection_set] == 1) / np.sum(ytrue == 1)
    else:
        fdp, tdp = 0, 0
    return fdp, tdp


# =========================
# CNN Model
# =========================
class BinaryConvNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =========================
# NN Wrapper
# =========================
class NNClassifier:
    def __init__(self, model, batch_size=64, n_epochs=10):
        self.model = model
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.n_epochs):
            for x, y in loader:
                self.model.train()
                y = y.float().unsqueeze(1)

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
# Custom subset
# =========================
class custom_subset(Dataset):
    def __init__(self, dataset, indices, labels=None):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        all_labels = torch.tensor(dataset.labels).squeeze()
        self.targets = all_labels[indices] if labels is None else labels

    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        target = self.targets[idx]
        return img, target

    def __len__(self):
        return len(self.targets)


# =========================
# Load MedMNIST
# =========================
data_flag = 'pathmnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

transform = transforms.ToTensor()
train_data = DataClass(split='train', transform=transform, download=True)

targets = torch.tensor(train_data.labels).squeeze()

print("Dataset size:", len(train_data))
print("Image shape:", train_data[0][0].shape)


# =========================
# Define null vs outliers
# =========================
inlr_labels = [0]
outlr_labels = [1]

inlr = (targets[..., None] == torch.tensor(inlr_labels)).any(-1).nonzero(as_tuple=True)[0]
outlr = (targets[..., None] == torch.tensor(outlr_labels)).any(-1).nonzero(as_tuple=True)[0]

print("Nulls:", len(inlr), "Outliers:", len(outlr))


# =========================
# Experiment settings
# =========================
level = 0.1
test_size = 1000
null_prop = 0.9

n_inlr = int(test_size * null_prop)
n_outlr = test_size - n_inlr

calib_size = 1000
train_size = 3000
n_runs = 5

fdp, tdp = [0.]*n_runs, [0.]*n_runs


# =========================
# Main loop
# =========================
for i in range(n_runs):

    # Test set
    test_idx = np.concatenate([
        np.random.choice(inlr, n_inlr, replace=False),
        np.random.choice(outlr, n_outlr, replace=False)
    ])

    test_dataset = custom_subset(train_data, test_idx, torch.ones(test_size))

    # Train / calibration split
    train_idx = np.setdiff1d(inlr, test_idx)
    np.random.shuffle(train_idx)

    calib_idx = train_idx[:calib_size]
    train_idx2 = train_idx[calib_size:calib_size+train_size]

    calib_dataset = custom_subset(train_data, calib_idx, torch.ones(calib_size))
    train_dataset = custom_subset(train_data, train_idx2, torch.zeros(train_size))

    full_train = ConcatDataset([train_dataset, calib_dataset, test_dataset])

    # Model
    n_channels = train_data[0][0].shape[0]
    model = BinaryConvNet(n_channels)
    clf = NNClassifier(model)

    clf.fit(full_train)

    # Scores
    test_scores = clf.predict_proba(test_dataset)
    null_scores = clf.predict_proba(calib_dataset)

    rej_set = EmpBH(null_scores, test_scores, level=level)

    # True labels
    test_labels = np.concatenate([np.zeros(n_inlr), np.ones(n_outlr)])

    fdp[i], tdp[i] = get_fdp(test_labels, rej_set)


print("FDP:", np.mean(fdp), np.std(fdp))
print("TDP:", np.mean(tdp), np.std(tdp))


# =========================
# Visualization
# =========================
idx_rej = rej_set
idx_acc = np.setdiff1d(np.arange(len(test_dataset)), idx_rej)

sample_rej = np.random.choice(idx_rej, size=min(10, len(idx_rej)), replace=False)
sample_acc = np.random.choice(idx_acc, size=min(14, len(idx_acc)), replace=False)

# Create figure with two sections
fig = plt.figure(figsize=(14, 10))

# Section 1: Outliers (Rejected)
fig.text(0.5, 0.98, 'Detected Outliers', ha='center', fontsize=14, fontweight='bold', color='red')
grid_outliers = ImageGrid(fig, 211, nrows_ncols=(2, 5), axes_pad=0.05)
images_rej = [test_dataset[i][0].permute(1,2,0).numpy() for i in sample_rej]

for ax, idx, im in zip(grid_outliers, sample_rej, images_rej):
    ax.imshow(im)
    # Check if correctly classified (true positive)
    is_correct = test_labels[idx] == 1
    if is_correct:
        ax.set_title('✓ Correct', fontsize=9, color='darkgreen', fontweight='bold')
    else:
        ax.set_title('✗ Wrong (FP)', fontsize=9, color='darkred', fontweight='bold')
    ax.axis('off')

# Section 2: Inliers (Accepted)
fig.text(0.5, 0.50, 'Normal Samples', ha='center', fontsize=14, fontweight='bold', color='green')
grid_inliers = ImageGrid(fig, 212, nrows_ncols=(2, 7), axes_pad=0.05)
images_acc = [test_dataset[i][0].permute(1,2,0).numpy() for i in sample_acc]

for ax, idx, im in zip(grid_inliers, sample_acc, images_acc):
    ax.imshow(im)
    # Check if correctly classified (true negative)
    is_correct = test_labels[idx] == 0
    if is_correct:
        ax.set_title('✓ Correct', fontsize=9, color='darkgreen', fontweight='bold')
    else:
        ax.set_title('✗ Wrong (FN)', fontsize=9, color='darkred', fontweight='bold')
    ax.axis('off')

plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4)
plt.savefig('outliers_visualization.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'outliers_visualization.png'")
plt.show()