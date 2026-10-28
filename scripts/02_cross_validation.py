"""Repeated stratified patient level cross validation of the whole pipeline.

Inside every fold, using training patients only: shortlist, QUBO, exact solver, QAOA at three
depths with five seeds each, and the classical methods. Panels are scored on the validation
patients of that fold with the organisers' fixed classifier. One JSON line is written per fold.
"""
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from joblib import Parallel, delayed

from qbio import SEED
from qbio.data import load_pool
from qbio.split import patient_folds

N_SPLITS, N_REPEATS, N_JOBS = 5, 10, 2
OUT = ROOT / "results" / "cv_folds.jsonl"


def to_python(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def work(fold):
    warnings.filterwarnings("ignore")
    from qbio import evaluate

    X, meta = load_pool()
    y, female, patients = meta["label"].to_numpy(), meta["female"].to_numpy(), meta.index.to_numpy()
    tr, va = fold["train"], fold["val"]
    t0 = time.perf_counter()
    rows, info, _ = evaluate.run_methods(X.iloc[tr], y[tr], female[tr])
    rows = evaluate.score_rows(rows, X, y, tr, va)
    oracle = evaluate.oracle_all_panels(X[info["shortlist"]], y, tr, va)
    return {
        "repeat": fold["repeat"],
        "fold": fold["fold"],
        "train_patients": patients[tr].tolist(),
        "val_patients": patients[va].tolist(),
        "info": info,
        "oracle": oracle,
        "rows": rows,
        "fold_runtime_s": time.perf_counter() - t0,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    X, meta = load_pool()
    folds = list(patient_folds(meta["label"], meta.index, N_SPLITS, N_REPEATS, SEED))
    print(f"{len(folds)} folds, {N_JOBS} workers", flush=True)
    OUT.write_text("")
    start = time.perf_counter()
    for i, result in enumerate(Parallel(n_jobs=N_JOBS, return_as="generator")(delayed(work)(f) for f in folds)):
        with OUT.open("a") as handle:
            handle.write(json.dumps(result, default=to_python) + "\n")
        print(f"fold {i + 1}/{len(folds)} done, {time.perf_counter() - start:.0f}s elapsed", flush=True)
