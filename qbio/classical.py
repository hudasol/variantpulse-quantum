"""Classical ways to choose a k gene panel from the same shortlist.

Every function works on the training patients only and returns a list of column positions
inside the shortlist, so all methods share the same data and the same shortlist.
"""
import itertools

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .shortlist import auroc_columns


def naive_single_gene(V, y, k):
    """Top k genes by single gene AUROC, ignoring redundancy (the starter notebook baseline)."""
    a = auroc_columns(V, y == 1)
    return [int(i) for i in np.argsort(-np.maximum(a, 1 - a))[:k]]


def greedy_mrmr(r, C, k):
    """Greedy forward mRMR on the same relevance r and redundancy C used in the QUBO."""
    n = len(r)
    chosen = [int(np.argmax(r))]
    while len(chosen) < k:
        rest = [j for j in range(n) if j not in chosen]
        score = [r[j] - C[j, chosen].mean() for j in rest]
        chosen.append(rest[int(np.argmax(score))])
    return chosen


def lasso_panel(V, y, k, seed):
    """L1 logistic regression forced to k genes.

    Walk up a grid of penalty strengths and take the first one that keeps exactly k genes.
    If the path jumps over k, take the k largest coefficients at the first strength with at
    least k genes.
    """
    Z = StandardScaler().fit_transform(V)
    fallback = None
    for c in np.logspace(-2.5, 1.5, 80):
        model = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=c, random_state=seed, max_iter=2000)
        model.fit(Z, y)
        coef = np.abs(model.coef_[0])
        nz = int((coef > 1e-8).sum())
        if nz == k:
            return [int(i) for i in np.flatnonzero(coef > 1e-8)]
        if nz > k and fallback is None:
            fallback = [int(i) for i in np.argsort(-coef)[:k]]
    return fallback if fallback is not None else [int(i) for i in np.argsort(-coef)[:k]]


def random_forest_panel(V, y, k, seed, n_trees=500):
    """Top k genes by random forest importance."""
    forest = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=1).fit(V, y)
    return [int(i) for i in np.argsort(-forest.feature_importances_)[:k]]


def simulated_annealing(Q, seed, n_steps=20000, t_start=10.0, t_end=0.02):
    """Single flip simulated annealing on the same QUBO matrix.

    Returns the lowest energy state visited, as a 0/1 vector, and the number of steps.
    The energy is diag(Q) . x + 0.5 * x^T S x, where S holds the pair terms mirrored.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    S = np.triu(Q, 1) + np.triu(Q, 1).T
    diag = np.diag(Q).copy()
    x = rng.integers(0, 2, size=n).astype(float)
    energy = float(diag @ x + 0.5 * x @ S @ x)
    best_x, best_e = x.copy(), energy
    temps = np.geomspace(t_start, t_end, n_steps)
    for t in temps:
        i = int(rng.integers(n))
        delta = (1 - 2 * x[i]) * (diag[i] + S[i] @ x)
        if delta <= 0 or rng.random() < np.exp(-delta / t):
            x[i] = 1 - x[i]
            energy += delta
            if energy < best_e - 1e-12:
                best_x, best_e = x.copy(), energy
    return best_x.astype(int), n_steps


def wrapper_exact_search(V, y, k, make_classifier, seed, n_splits=5):
    """Check every k gene panel with the fixed classifier inside the training patients.

    Each panel is scored by out of fold AUROC over a stratified split of the training patients.
    Many panels tie at AUROC 1.0 on so few patients, so ties are broken by the lower out of
    fold log loss. This is an exact search, but over a noisy criterion.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(V, y))
    best, best_key = None, None
    n = V.shape[1]
    for panel in itertools.combinations(range(n), k):
        cols = list(panel)
        oof = np.zeros(len(y))
        for tr, va in folds:
            model = make_classifier().fit(V[np.ix_(tr, cols)], y[tr])
            oof[va] = model.predict_proba(V[np.ix_(va, cols)])[:, 1]
        key = (roc_auc_score(y, oof), -log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)))
        if best_key is None or key > best_key:
            best, best_key = cols, key
    return best, best_key
