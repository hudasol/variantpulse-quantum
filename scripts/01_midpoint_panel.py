"""Midpoint: choose the four gene panel from all patients of the team pool.

Shortlist, QUBO and exact solver on all 26 patients, then QAOA in Qiskit at three depths with
ten seeds each. The panel is the one most often returned as the lowest energy valid panel
among the final samples of the thirty QAOA runs. Every run is written to a results log.
Validation of the whole procedure is done separately in 02_cross_validation.py, where the
shortlist and the panel are chosen inside training patients only.
"""
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
import qiskit_algorithms
import qiskit_optimization
import scipy
import sklearn

from qbio import PANEL_SIZE, SEED, qaoa, qubo
from qbio.data import load_pool
from qbio.shortlist import make_shortlist

CFG = {"k": PANEL_SIZE, "k_short": 16, "reps": (1, 2, 3), "seeds": tuple(range(10)),
       "maxiter": 150, "shots": 2048, "final_shots": 8192}
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
(ROOT / "midpoint").mkdir(exist_ok=True)

X, meta = load_pool()
y, female = meta["label"].to_numpy(), meta["female"].to_numpy()

sl = make_shortlist(X, y, female, k=CFG["k_short"])
genes = sl["genes"]
V = X.iloc[:, sl["columns"]].to_numpy(dtype=float)
P = qubo.make_problem(genes, V, sl["t"], k=CFG["k"])
ex = qubo.exhaustive(P)
E, sizes = ex["energies"], ex["sizes"]
opt_states = [int(s) for s in ex["optimum_states"]]
n = P.n

shortlist_table = pd.DataFrame({
    "gene": genes,
    "welch_t": sl["t"],
    "higher_in": np.where(sl["t"] > 0, "tumour", "normal"),
    "relevance": P.r,
    "mean_redundancy": P.C.sum(axis=1) / (n - 1),
})
shortlist_table.to_csv(OUT / "shortlist_full_pool.csv", index=False)

rows = []
for reps in CFG["reps"]:
    for seed in CFG["seeds"]:
        run = qaoa.run_qaoa(P.Q, E, reps, seed, CFG["maxiter"], CFG["shots"], CFG["final_shots"])
        s = qaoa.summarise(run, E, sizes, CFG["k"], opt_states)
        state = s["best_sampled_valid_state"]
        rows.append({
            "method": "qaoa", "library": "qiskit", "reps": reps, "seed": seed,
            "settings": f"COBYLA maxiter={CFG['maxiter']}, {CFG['shots']} shots per evaluation, {CFG['final_shots']} final shots",
            "circuit_runs": run["circuit_runs"], "shots_total": run["shots_total"],
            "runtime_s": round(run["runtime_s"], 2),
            "prob_right_size": round(s["prob_right_size"], 4),
            "prob_optimum": round(s["prob_optimum"], 5),
            "most_probable_size": s["most_probable_size"],
            "found_optimum": s["found_optimum"],
            "best_valid_energy": None if state is None else round(float(E[state]), 4),
            "panel": None if state is None else ";".join(P.panel(qubo.state_bits(state, n))),
        })

runs = pd.DataFrame(rows)
runs.to_csv(OUT / "midpoint_runs.csv", index=False)

valid = runs.dropna(subset=["panel"])
votes = Counter(valid["panel"])
energy_of = {p: float(g["best_valid_energy"].min()) for p, g in valid.groupby("panel")}
chosen, _ = sorted(votes.items(), key=lambda kv: (-kv[1], energy_of[kv[0]]))[0]
panel = chosen.split(";")

exact_panel = [genes[i] for i, b in enumerate(qubo.state_bits(opt_states[0], n)) if b]
size4 = np.sort(E[sizes == CFG["k"]])
summary = {
    "seed_base": SEED,
    "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
    "versions": {"qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
                 "qiskit_optimization": qiskit_optimization.__version__,
                 "qiskit_algorithms": qiskit_algorithms.__version__, "scikit_learn": sklearn.__version__,
                 "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
    "patients": int(len(y)), "tumour": int(y.sum()), "normal": int((y == 0).sum()),
    "shortlist_log": {k: v for k, v in sl["log"].items() if k != "sex_linked_genes"},
    "sex_linked_genes_removed": sl["log"]["sex_linked_genes"],
    "shortlist": genes,
    "qubo": {"alpha": P.alpha, "beta": round(P.beta, 4), "lam_min": round(P.lam_min, 4), "lam": round(P.lam, 4),
             "states": ex["n_states"], "exact_value": round(ex["best_value"], 4),
             "exact_panel": exact_panel, "n_tied_optima": len(opt_states),
             "gap_best_to_second_size4": round(float(size4[1] - size4[0]), 5)},
    "qaoa_runs": int(len(runs)),
    "qaoa_found_exact_optimum_runs": int(runs["found_optimum"].sum()),
    "qaoa_votes": dict(votes),
    "qaoa_mean_prob_right_size": round(float(runs["prob_right_size"].mean()), 4),
    "qaoa_runs_where_most_probable_string_has_right_size": int((runs["most_probable_size"] == CFG["k"]).sum()),
    "qaoa_total_runtime_s": round(float(runs["runtime_s"].sum()), 1),
    "panel": panel,
    "panel_equals_exact_optimum": sorted(panel) == sorted(exact_panel),
}
(OUT / "midpoint_summary.json").write_text(json.dumps(summary, indent=2, default=str))
pd.DataFrame({"gene": sorted(panel)}).to_csv(ROOT / "midpoint" / "panel.csv", index=False)

print(json.dumps({k: summary[k] for k in ("shortlist_log", "qubo", "qaoa_runs", "qaoa_found_exact_optimum_runs",
                                           "qaoa_votes", "qaoa_mean_prob_right_size",
                                           "qaoa_runs_where_most_probable_string_has_right_size",
                                           "qaoa_total_runtime_s", "panel", "panel_equals_exact_optimum")}, indent=2))
print(shortlist_table.round(3).to_string(index=False))
print(runs.groupby("reps").agg(found_opt=("found_optimum", "sum"), p_size4=("prob_right_size", "mean"),
                              runs=("circuit_runs", "mean"), runtime=("runtime_s", "mean")).round(3))
