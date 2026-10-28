"""QUBO for choosing a panel of k genes from a shortlist.

Each shortlisted gene i is one binary variable x_i (1 means the gene is in the panel).
The value to minimise is

    E(x) = - alpha * sum_i r_i x_i  +  beta * sum_{i<j} c_ij x_i x_j  +  lam * (sum_i x_i - k)^2

r_i is the relevance of gene i, scaled to (0, 1] (absolute Welch t statistic divided by the
largest in the shortlist). c_ij is the redundancy of a pair, the absolute Pearson correlation
between the two genes across training patients. alpha = 1 and beta = 2 / (k - 1), which gives
the mean relevance and the mean redundancy of a k gene panel the same weight, as in mRMR.
The penalty weight lam is tuned from training data only (see `tune_lambda`).

As a matrix, E(x) = x^T Q x + lam * k^2, so the constant lam * k^2 is left out of Q.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class Problem:
    genes: list
    r: np.ndarray
    C: np.ndarray
    k: int
    alpha: float
    beta: float
    lam: float
    lam_min: float
    Q: np.ndarray

    @property
    def n(self):
        return len(self.genes)

    def panel(self, bits):
        """Gene names selected by a 0/1 vector."""
        return [g for g, b in zip(self.genes, bits) if b]


def relevance(t):
    a = np.abs(np.asarray(t, dtype=float))
    return a / a.max()


def redundancy(V):
    C = np.abs(np.corrcoef(V, rowvar=False))
    np.fill_diagonal(C, 0.0)
    return C


@lru_cache(maxsize=4)
def all_states(n):
    """All 2^n bit vectors as a float32 matrix, bit i of row s is variable i."""
    idx = np.arange(2**n, dtype=np.int64)[:, None]
    return ((idx >> np.arange(n)) & 1).astype(np.float32)


def base_energy(r, C, alpha, beta, B):
    """Relevance and redundancy part of the energy for every state in B."""
    U = np.triu(C, 1).astype(np.float32)
    return -alpha * (B @ r.astype(np.float32)) + beta * ((B @ U) * B).sum(axis=1)


def lambda_min(r, C, k, alpha=1.0, beta=None):
    """Smallest penalty weight for which the global optimum of the QUBO has exactly k genes."""
    beta = 2.0 / (k - 1) if beta is None else beta
    n = len(r)
    B = all_states(n)
    s = B.sum(axis=1).astype(int)
    base = base_energy(r, C, alpha, beta, B)
    best = np.array([base[s == size].min() for size in range(n + 1)])
    need = [(best[k] - best[size]) / (size - k) ** 2 for size in range(n + 1) if size != k]
    return float(max(max(need), 0.0))


def build_Q(r, C, k, lam, alpha=1.0, beta=None):
    beta = 2.0 / (k - 1) if beta is None else beta
    n = len(r)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -alpha * r[i] + lam * (1 - 2 * k)
        for j in range(i + 1, n):
            Q[i, j] = beta * C[i, j] + 2 * lam
    return Q


def make_problem(genes, V, t, k=4, margin=2.0, lam_floor=0.5, alpha=1.0):
    """Build the QUBO from training data V (patients x shortlist genes) and statistics t."""
    r = relevance(t)
    C = redundancy(V)
    beta = 2.0 / (k - 1)
    lam_lo = lambda_min(r, C, k, alpha, beta)
    lam = max(margin * lam_lo, lam_floor)
    return Problem(list(genes), r, C, k, alpha, beta, lam, lam_lo, build_Q(r, C, k, lam, alpha, beta))


def energies(Q, B):
    """x^T Q x for every row of B."""
    Bf = B.astype(np.float32)
    return ((Bf @ Q.astype(np.float32)) * Bf).sum(axis=1)


def exhaustive(problem):
    """Check every one of the 2^n choices. Returns the exact optimum and all energies."""
    B = all_states(problem.n)
    E = energies(problem.Q, B)
    size = B.sum(axis=1).astype(int)
    best = E.min()
    optimum = np.flatnonzero(np.isclose(E, best, atol=1e-5))
    ok = np.flatnonzero(size == problem.k)
    best_k = E[ok].min()
    return {
        "energies": E,
        "sizes": size,
        "best_value": float(best),
        "optimum_states": optimum,
        "optimum_sizes": size[optimum].tolist(),
        "best_value_size_k": float(best_k),
        "best_k_states": ok[np.isclose(E[ok], best_k, atol=1e-5)],
        "n_states": int(len(E)),
    }


def state_bits(state, n):
    return [(int(state) >> i) & 1 for i in range(n)]
