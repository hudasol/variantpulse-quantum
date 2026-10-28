"""Run every method on one training split and score the panels on the validation patients.

All methods see the same training patients, the same shortlist and the same relevance and
redundancy numbers. Panels are scored with the organisers' fixed classifier (panel_checker).
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import classical, qaoa, qubo
from .shortlist import make_shortlist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external"))
from panel_checker import make_classifier, validate_panel  # noqa: E402  (organisers' file)

CONFIG = {
    "k": 4,
    "k_short": 16,
    "reps": (1, 2, 3),
    "seeds": (0, 1, 2, 3, 4),
    "maxiter": 150,
    "shots": 2048,
    "final_shots": 8192,
    "sa_steps": 20000,
    "wrapper": True,
}


def random_sampling(E, sizes, k, n_samples, rng):
    """Draw uniform random bit strings and keep the lowest energy one of the right size."""
    s = rng.integers(0, len(E), size=n_samples)
    valid = s[sizes[s] == k]
    if len(valid) == 0:
        return None
    return int(valid[np.argmin(E[valid])])


def run_methods(X, y, female, cfg=CONFIG):
    """Shortlist, build the QUBO and run every method on the TRAINING patients (X, y, female)."""
    k = cfg["k"]
    sl = make_shortlist(X, y, female, k=cfg["k_short"])
    genes = sl["genes"]
    V = X.iloc[:, sl["columns"]].to_numpy(dtype=float)
    P = qubo.make_problem(genes, V, sl["t"], k=k)
    ex = qubo.exhaustive(P)
    E, sizes = ex["energies"], ex["sizes"]
    opt_states = [int(s) for s in ex["optimum_states"]]
    n = P.n
    rows = []

    def add(method, variant, seed, state=None, idx=None, runtime=None, **extra):
        if state is not None:
            idx = [i for i, b in enumerate(qubo.state_bits(state, n)) if b]
        panel = None if idx is None else [genes[i] for i in sorted(idx)]
        rows.append(
            dict(method=method, variant=variant, seed=seed, panel=panel, runtime_s=runtime, **extra)
        )

    t0 = time.perf_counter()
    ex2 = qubo.exhaustive(P)
    add("exact_qubo", "all_2^n_states", 0, state=opt_states[0], runtime=time.perf_counter() - t0,
        found_optimum=True, states_checked=ex2["n_states"], n_tied_optima=len(opt_states))

    for reps in cfg["reps"]:
        for seed in cfg["seeds"]:
            run = qaoa.run_qaoa(P.Q, E, reps, seed, cfg["maxiter"], cfg["shots"], cfg["final_shots"])
            s = qaoa.summarise(run, E, sizes, k, opt_states)
            state = s["best_sampled_valid_state"]
            add("qaoa", f"p={reps}", seed, state=state, runtime=run["runtime_s"],
                found_optimum=s["found_optimum"], prob_right_size=s["prob_right_size"],
                prob_optimum=s["prob_optimum"], most_probable_size=s["most_probable_size"],
                circuit_runs=run["circuit_runs"], shots_total=run["shots_total"],
                final_mean_energy=run["final_mean_energy"], reps=reps)
            for variant, n_samples in (("same_final_shots", cfg["final_shots"]), ("same_total_shots", run["shots_total"])):
                rng = np.random.default_rng(10_000 + 100 * reps + seed)
                t1 = time.perf_counter()
                st = random_sampling(E, sizes, k, n_samples, rng)
                add("random_sampling", f"{variant} (p={reps})", seed, state=st, runtime=time.perf_counter() - t1,
                    found_optimum=bool(st is not None and st in opt_states), shots_total=n_samples, reps=reps)

    for seed in cfg["seeds"]:
        t1 = time.perf_counter()
        x, steps = classical.simulated_annealing(P.Q, seed, cfg["sa_steps"])
        size = int(x.sum())
        state = int(sum(int(b) << i for i, b in enumerate(x))) if size == k else None
        add("simulated_annealing", f"{cfg['sa_steps']} steps", seed, state=state, runtime=time.perf_counter() - t1,
            found_optimum=bool(state is not None and state in opt_states), best_size=size)

    yv = np.asarray(y)
    t1 = time.perf_counter()
    add("naive_single_gene", "top 4 by AUROC", 0, idx=classical.naive_single_gene(V, yv, k), runtime=time.perf_counter() - t1)
    t1 = time.perf_counter()
    add("greedy_mrmr", "same relevance and redundancy", 0, idx=classical.greedy_mrmr(P.r, P.C, k), runtime=time.perf_counter() - t1)
    for seed in cfg["seeds"]:
        t1 = time.perf_counter()
        add("lasso", "L1 logistic, forced to 4", seed, idx=classical.lasso_panel(V, yv, k, seed), runtime=time.perf_counter() - t1)
        t1 = time.perf_counter()
        add("random_forest", "500 trees", seed, idx=classical.random_forest_panel(V, yv, k, seed), runtime=time.perf_counter() - t1)
    if cfg["wrapper"]:
        t1 = time.perf_counter()
        idx, key = classical.wrapper_exact_search(V, yv, k, make_classifier, seed=0)
        add("wrapper_exact", "all 1820 panels, inner CV", 0, idx=idx, runtime=time.perf_counter() - t1,
            inner_auroc=key[0])

    info = {
        "shortlist": genes,
        "shortlist_log": sl["log"],
        "lam": P.lam,
        "lam_min": P.lam_min,
        "exact_value": ex["best_value"],
        "n_tied_optima": len(opt_states),
        "exact_panel": [genes[i] for i, b in enumerate(qubo.state_bits(opt_states[0], n)) if b],
        "second_best_gap": float(np.sort(E[sizes == k])[1] - np.sort(E[sizes == k])[0]),
    }
    return rows, info, V


def score_rows(rows, X, y, tr, va):
    """Add validation AUROC (organisers' classifier) to every row that has a valid panel."""
    for r in rows:
        if r["panel"] is None:
            r["auroc"] = np.nan
        else:
            r["auroc"] = validate_panel(r["panel"], X.iloc[tr], y[tr], X.iloc[va], y[va])
    return rows


def oracle_all_panels(Xs, y, tr, va, k=4):
    """AUROC of every k gene panel of the shortlist on validation patients (context only)."""
    out = []
    A, B = Xs.to_numpy()[tr], Xs.to_numpy()[va]
    for panel in itertools.combinations(range(Xs.shape[1]), k):
        cols = list(panel)
        model = make_classifier().fit(A[:, cols], y[tr])
        out.append(roc_auc_score(y[va], model.predict_proba(B[:, cols])[:, 1]))
    out = np.array(out)
    return {"oracle_max": float(out.max()), "all_panels_mean": float(out.mean()),
            "share_ge_095": float((out >= 0.95).mean()), "n_panels": int(len(out))}
