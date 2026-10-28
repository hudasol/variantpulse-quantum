"""Summarise the repeated cross validation written by 02_cross_validation.py.

Reads results/cv_folds.jsonl and writes results/cv_methods.csv and results/cv_summary.json.
Per method the numbers are first averaged over seeds inside a fold, then over folds. Besides the
validation AUROC (organisers' fixed classifier) it computes the validation log loss from the saved
panels, because AUROC is close to 1.0 for most methods on so few patients and cannot separate them.
"""
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
from sklearn.metrics import log_loss
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
        records.append({
            "repeat": fold["repeat"], "fold": fold["fold"], "method": label,
            "auroc": row.get("auroc"), "log_loss": loss,
            "has_panel": row.get("panel") is not None,
            "found_optimum": row.get("found_optimum"),
            "prob_right_size": row.get("prob_right_size"),
            "top_bitstring_right_size": None if row.get("most_probable_size") is None else row["most_probable_size"] == 4,
        })
d = pd.DataFrame(records)
per_fold = d.groupby(["method", "repeat", "fold"]).agg(
    auroc=("auroc", "mean"), log_loss=("log_loss", "mean"), found=("found_optimum", "mean"),
    has_panel=("has_panel", "mean"), p_size=("prob_right_size", "mean"), top_ok=("top_bitstring_right_size", "mean")).reset_index()
table = per_fold.groupby("method").agg(
    folds=("auroc", "size"),
    auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
    log_loss_mean=("log_loss", "mean"), log_loss_sd=("log_loss", "std"),
    found_optimum=("found", "mean"), returned_a_valid_panel=("has_panel", "mean"),
    prob_right_size=("p_size", "mean"), top_bitstring_right_size=("top_ok", "mean")).round(4)
order = ["exact_qubo", "qaoa, p=1", "qaoa, p=2", "qaoa, p=3", "simulated_annealing",
         "random_sampling, same_final_shots", "random_sampling, same_total_shots",
         "greedy_mrmr", "naive_single_gene", "lasso", "random_forest", "wrapper_exact"]
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
}
(ROOT / "results" / "cv_summary.json").write_text(json.dumps(summary, indent=2))
print(f"{n} folds")
print(table.to_string())
print(json.dumps(summary, indent=2))
