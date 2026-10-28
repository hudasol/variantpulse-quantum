"""Summarise the repeated cross validation written by 02_cross_validation.py.

Reads results/cv_folds.jsonl and writes results/cv_methods.csv and results/cv_summary.json.
Per method the numbers are first averaged over seeds inside a fold, then over folds. Besides the
validation AUROC (organisers' fixed classifier) it computes the validation log loss from the saved
panels, because AUROC is close to 1.0 for most methods on so few patients and cannot separate them.
It also scores every four gene panel of each fold's shortlist (the expected result of picking a
panel at random, cached in results/cv_random_panels.csv), pools the three QAOA depths, and gives
95% bootstrap intervals over folds for the share of runs that found the exact optimum. The folds
overlap (they reshuffle the same 26 patients), so the intervals understate the uncertainty.
"""
import itertools
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "external"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score
from panel_checker import make_classifier

from qbio.data import load_pool

SUBMITTED = json.loads((ROOT / "results" / "midpoint_summary.json").read_text())["panel"]
folds = [json.loads(line) for line in (ROOT / "results" / "cv_folds.jsonl").read_text().splitlines() if line.strip()]
X, meta = load_pool()
y = pd.Series(meta["label"].to_numpy(), index=meta.index)

records = []
for fold in folds:
    tr, va = fold["train_patients"], fold["val_patients"]
    for row in fold["rows"]:
        label = row["method"] if row["method"] != "random_sampling" else f"random_sampling, {row['variant'].split(' (')[0]}"
        if row["method"] == "qaoa":
            label = f"qaoa, {row['variant']}"
        loss = np.nan
        if row.get("panel"):
            model = make_classifier().fit(X.loc[tr, row["panel"]].to_numpy(float), y.loc[tr].to_numpy())
            p = model.predict_proba(X.loc[va, row["panel"]].to_numpy(float))[:, 1]
            loss = log_loss(y.loc[va].to_numpy(), np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
        # Same panel as the exact optimum of the fold's QUBO. QUBO methods log this themselves;
        # for the other methods it is the plain comparison of gene sets.
        same = row.get("found_optimum")
        if same is None and row.get("panel"):
            same = set(row["panel"]) == set(fold["info"]["exact_panel"])
        records.append({
            "repeat": fold["repeat"], "fold": fold["fold"], "method": label,
            "auroc": row.get("auroc"), "log_loss": loss,
            "has_panel": row.get("panel") is not None,
            "found_optimum": same,
            "prob_right_size": row.get("prob_right_size"),
            "prob_optimum": row.get("prob_optimum"),
            "top_bitstring_right_size": None if row.get("most_probable_size") is None else row["most_probable_size"] == 4,
        })
d = pd.DataFrame(records)
per_fold = d.groupby(["method", "repeat", "fold"]).agg(
    auroc=("auroc", "mean"), log_loss=("log_loss", "mean"), found=("found_optimum", "mean"),
    has_panel=("has_panel", "mean"), p_size=("prob_right_size", "mean"), p_opt=("prob_optimum", "mean"),
    top_ok=("top_bitstring_right_size", "mean")).reset_index()
# QAOA pooled over the three depths, one value per fold
pooled = per_fold[per_fold["method"].str.startswith("qaoa, p=")].groupby(["repeat", "fold"]).mean(numeric_only=True).reset_index()
pooled["method"] = "qaoa, all depths"
per_fold = pd.concat([per_fold, pooled], ignore_index=True)
table = per_fold.groupby("method").agg(
    folds=("auroc", "size"),
    auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
    log_loss_mean=("log_loss", "mean"), log_loss_sd=("log_loss", "std"),
    found_optimum=("found", "mean"), returned_a_valid_panel=("has_panel", "mean"),
    prob_right_size=("p_size", "mean"), prob_optimum=("p_opt", "mean"),
    top_bitstring_right_size=("top_ok", "mean"))
DIGITS = lambda t: {c: (6 if c == "prob_optimum" else 4) for c in t.columns}  # the optimum holds a tiny share of shots

# 95% bootstrap interval over folds for the share of runs that found the exact optimum
rng = np.random.default_rng(0)
for name, values in per_fold.groupby("method")["found"].apply(lambda s_: s_.to_numpy(float)).items():
    means = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    table.loc[name, "found_lo"], table.loc[name, "found_hi"] = np.percentile(means, [2.5, 97.5]).round(4)


def score_all_panels(fold):
    """Mean validation AUROC and log loss over every four gene panel of the fold's shortlist."""
    warnings.filterwarnings("ignore")
    cols, tr, va = fold["info"]["shortlist"], fold["train_patients"], fold["val_patients"]
    A, B = X.loc[tr, cols].to_numpy(float), X.loc[va, cols].to_numpy(float)
    ytr, yva = y.loc[tr].to_numpy(), y.loc[va].to_numpy()
    aucs, losses = [], []
    for panel in itertools.combinations(range(len(cols)), 4):
        c = list(panel)
        p = make_classifier().fit(A[:, c], ytr).predict_proba(B[:, c])[:, 1]
        aucs.append(roc_auc_score(yva, p))
        losses.append(log_loss(yva, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    return {"repeat": fold["repeat"], "fold": fold["fold"], "auroc": float(np.mean(aucs)), "log_loss": float(np.mean(losses))}


CACHE = ROOT / "results" / "cv_random_panels.csv"
if CACHE.exists() and len(pd.read_csv(CACHE)) == len(folds):
    rand = pd.read_csv(CACHE)
else:
    rand = pd.DataFrame(Parallel(n_jobs=2)(delayed(score_all_panels)(f) for f in folds))
    rand.to_csv(CACHE, index=False)
table.loc["random_four_gene_panel", ["folds", "auroc_mean", "auroc_sd", "log_loss_mean", "log_loss_sd"]] = [
    len(rand), rand["auroc"].mean(), rand["auroc"].std(), rand["log_loss"].mean(), rand["log_loss"].std()]
table = table.round(DIGITS(table))

order = ["exact_qubo", "qaoa, all depths", "qaoa, p=1", "qaoa, p=2", "qaoa, p=3", "simulated_annealing",
         "random_sampling, same_final_shots", "random_sampling, same_total_shots",
         "greedy_mrmr", "naive_single_gene", "lasso", "random_forest", "wrapper_exact", "random_four_gene_panel"]
table = table.reindex([m for m in order if m in table.index] + [m for m in table.index if m not in order])
table.to_csv(ROOT / "results" / "cv_methods.csv")

exact_freq, short_freq, panel_counts = Counter(), Counter(), Counter()
for fold in folds:
    exact_freq.update(fold["info"]["exact_panel"])
    short_freq.update(fold["info"]["shortlist"])
    panel_counts[";".join(sorted(fold["info"]["exact_panel"]))] += 1
n = len(folds)
oracle = pd.DataFrame([f["oracle"] for f in folds]).mean().round(4).to_dict()
summary = {
    "folds": n,
    "submitted_panel": SUBMITTED,
    "submitted_gene_in_fold_shortlist": {g: short_freq[g] for g in SUBMITTED},
    "submitted_gene_in_fold_exact_panel": {g: exact_freq[g] for g in SUBMITTED},
    "most_frequent_in_fold_exact_panel": exact_freq.most_common(8),
    "distinct_fold_exact_panels": len(panel_counts),
    "fold_runtime_median_s": round(float(np.median([f["fold_runtime_s"] for f in folds])), 1),
    "all_1820_panels_oracle_mean": oracle,
    "random_panel_auroc": round(float(rand["auroc"].mean()), 4),
    "random_panel_log_loss": round(float(rand["log_loss"].mean()), 4),
}
(ROOT / "results" / "cv_summary.json").write_text(json.dumps(summary, indent=2))
print(f"{n} folds")
print(table.to_string())
print(json.dumps(summary, indent=2))
