import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from mpl_toolkits.axes_grid1 import make_axes_locatable

RNG = np.random.default_rng(30)

# parameters
n = 3000
m0 = 500
m1 = 500
m = m0 + m1
k = 2000
ell = n - k

def mvn_pdf(x, mean, cov_scale=1.0):
    mean = np.asarray(mean)
    d = mean.size
    diff = x - mean
    return (2 * np.pi * cov_scale**2) ** (-d / 2) * np.exp(
        -0.5 * np.sum(diff**2, axis=1) / cov_scale**2
    )

def f0_density(x):
    return mvn_pdf(x, mean=[0.0, 0.0], cov_scale=1.0)

def f1_density(x, two_sided=False):
    if two_sided:
        return (
            0.5 * mvn_pdf(x, mean=[2.0, 2.0], cov_scale=1.0)
            + 0.5 * mvn_pdf(x, mean=[-2.0, -2.0], cov_scale=1.0)
        )
    else:
        return mvn_pdf(x, mean=[2.0, 2.0], cov_scale=1.0)

def sample_2d_pu(two_sided=False):
    Y = RNG.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(n, 2))

    X_0 = RNG.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(m0, 2))
    if two_sided:
        mix = RNG.binomial(1, 0.5, size=m1)
        means = np.where(mix[:, None] == 1, [2.0, 2.0], [-2.0, -2.0])
        X_1 = RNG.normal(loc=means, scale=[1.0, 1.0], size=(m1, 2))
    else:
        X_1 = RNG.normal(loc=[2.0, 2.0], scale=[1.0, 1.0], size=(m1, 2))

    Z = np.vstack([Y, X_0, X_1])

    y_true = np.concatenate([
        np.zeros(n, dtype=int),
        np.zeros(m0, dtype=int),
        np.ones(m1, dtype=int)
    ])

    # PU labels:
    # -1 = labeled null sample Y_1,...,Y_k
    # +1 = mixed sample Y_{k+1},...,Y_n, X_1,...,X_m
    y_pu = np.concatenate([
        np.full(k, -1, dtype=int),
        np.ones(n + m - k, dtype=int)
    ])
    return Z, y_true, y_pu

# vælg her
two_sided = True
Z, y_true, y_pu = sample_2d_pu(two_sided=two_sided)

# fit modeller
logreg = LogisticRegression(max_iter=2000)
logreg.fit(Z, y_pu)

nn = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=0
)
nn.fit(Z, y_pu)

# grid
x1_min, x1_max = Z[:, 0].min() - 1, Z[:, 0].max() + 1
x2_min, x2_max = Z[:, 1].min() - 1, Z[:, 1].max() + 1

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)
grid = np.c_[xx1.ravel(), xx2.ravel()]

# learned scores \hat g
score_logreg = logreg.predict_proba(grid)[:, 1].reshape(xx1.shape)
score_nn = nn.predict_proba(grid)[:, 1].reshape(xx1.shape)

# population score g^sharp
gamma = m1 / (ell + m)

f0_grid = f0_density(grid)
f1_grid = f1_density(grid, two_sided=two_sided)
f_gamma_grid = (1 - gamma) * f0_grid + gamma * f1_grid

g_sharp = ((ell + m) * f_gamma_grid) / ((ell + m) * f_gamma_grid + k * f0_grid)
g_sharp = g_sharp.reshape(xx1.shape)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# true labels
axes[0].scatter(Z[y_true == 0, 0], Z[y_true == 0, 1], c="#fe7a06", s=8, alpha=0.7, label="$X \sim P_0$")
axes[0].scatter(Z[y_true == 1, 0], Z[y_true == 1, 1], c="#598dec", s=8, alpha=0.7, label="$X \sim P_1$")
#axes[0].set_title("True labels")
axes[0].legend(loc="upper left")


# PU labels
axes[1].scatter(Z[y_pu == -1, 0], Z[y_pu == -1, 1], c="#fe7a06", s=8, alpha=0.7, label="Positive sample")
axes[1].scatter(Z[y_pu == 1, 0], Z[y_pu == 1, 1], c="#598dec", s=8, alpha=0.7, label="Unlabeled sample")
#axes[1].set_title("PU labels")
axes[1].legend(loc="upper left")

# g^sharp reference under logistic regression
c2 = axes[2].contourf(xx1, xx2, g_sharp, levels=12, cmap="RdBu", alpha=0.9)
axes[2].scatter(Z[y_true == 0, 0], Z[y_true == 0, 1], c="#fe7a06", s=8, alpha=0.7, label="$X \sim P_0$")
axes[2].scatter(Z[y_true == 1, 0], Z[y_true == 1, 1], c="#598dec", s=8, alpha=0.7, label="$X \sim P_1$")
#axes[2].set_title(r"$g^\sharp$")
divider = make_axes_locatable(axes[2])
cax = divider.append_axes("right", size="5%", pad=0.05) 
fig.colorbar(c2, cax=cax)
axes[2].legend(loc="upper left")

# Logistic regression \hat g
c0 = axes[3].contourf(xx1, xx2, score_logreg, levels=12, cmap="RdBu", alpha=0.9)
axes[3].scatter(Z[y_true == 0, 0], Z[y_true == 0, 1], c="#fe7a06", s=8, alpha=0.7, label="$X \sim P_0$")
axes[3].scatter(Z[y_true == 1, 0], Z[y_true == 1, 1], c="#598dec", s=8, alpha=0.7, label="$X \sim P_1$")
#axes[3].set_title(r"$\hat g$ (Logistic regression)")
divider = make_axes_locatable(axes[3])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(c0, cax=cax)
axes[3].legend(loc="upper left")

# Neural network \hat g
c1 = axes[4].contourf(xx1, xx2, score_nn, levels=12, cmap="RdBu", alpha=0.9)
axes[4].scatter(Z[y_true == 0, 0], Z[y_true == 0, 1], c="#fe7a06", s=8, alpha=0.7, label="$X \sim P_0$")
axes[4].scatter(Z[y_true == 1, 0], Z[y_true == 1, 1], c="#598dec", s=8, alpha=0.7, label="$X \sim P_1$")
#axes[4].set_title(r"$\hat g$ (Neural network)")
divider = make_axes_locatable(axes[4])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(c1, cax=cax)
axes[4].legend(loc="upper left")

for ax in axes.ravel():
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

plt.tight_layout()
fig.savefig(f"{two_sided}_figur4_with_gsharp.pdf", format="pdf", bbox_inches="tight")
plt.show()
