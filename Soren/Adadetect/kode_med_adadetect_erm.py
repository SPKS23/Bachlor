import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from procedure import AdaDetectERM

RNG = np.random.default_rng(0)

# parameters
n = 5000
m0 = 900
m1 = 100
m = m0 + m1
k = 4000
level = 0.05

two_sided = True 

def get_fdp(ytrue, rejection_set):
    if rejection_set.size:
        fdp = np.sum(ytrue[rejection_set] == 0) / len(rejection_set)
        tdp = np.sum(ytrue[rejection_set] == 1) / np.sum(ytrue == 1)
    else:
        fdp = 0
        tdp = 0
    return fdp, tdp

def sample_2d_pu(two_sided=two_sided):
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
    y_pu = np.concatenate([
        np.full(k, -1, dtype=int),
        np.ones(n + m - k, dtype=int)
    ])
    return Z, y_true, y_pu

def run_once_on_data(model, Z, y_true):
    xnull = Z[:n]
    x = Z[n:n+m]
    y_test_true = y_true[n:n+m]

    proc = AdaDetectERM(
        scoring_fn=model,
        correction_type="storey",
        split_size=k / n
    )

    rejection_set = proc.apply(x, level, xnull)
    return get_fdp(y_test_true, rejection_set), len(rejection_set)

B = 5

fdps_logreg, tdps_logreg, rejs_logreg = [], [], []
fdps_nn, tdps_nn, rejs_nn = [], [], []

for b in range(B):
    Z, y_true, y_pu = sample_2d_pu(two_sided=two_sided)

    (fdp, tdp), r = run_once_on_data(
        LogisticRegression(max_iter=2000),
        Z, y_true
    )
    fdps_logreg.append(fdp)
    tdps_logreg.append(tdp)
    rejs_logreg.append(r)

    (fdp, tdp), r = run_once_on_data(
        MLPClassifier(
            hidden_layer_sizes=(50, 50),
            activation="relu",
            max_iter=2000,
            random_state=b
        ),
        Z, y_true
    )
    fdps_nn.append(fdp)
    tdps_nn.append(tdp)
    rejs_nn.append(r)

print("Logistic regression:")
print("Mean FDP (empirical FDR estimate):", np.mean(fdps_logreg))
print("Mean TDP:", np.mean(tdps_logreg))
print("Mean number of rejections:", np.mean(rejs_logreg))

print("\nNeural network:")
print("Mean FDP (empirical FDR estimate):", np.mean(fdps_nn))
print("Mean TDP:", np.mean(tdps_nn))
print("Mean number of rejections:", np.mean(rejs_nn))
