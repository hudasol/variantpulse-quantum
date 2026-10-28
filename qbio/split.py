"""Patient level splits.

Every patient has one sample here, but the splitter is still grouped by patient so the rule
"one patient, one group" holds by construction and keeps holding if samples are ever added.
Folds are stratified so that both tumour and normal patients appear in training and validation.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def patient_folds(y, groups, n_splits=5, n_repeats=10, seed=42):
    """Yield repeated stratified folds as dicts with repeat, fold, train and val indices."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    for rep in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for k, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            assert set(groups[tr]).isdisjoint(groups[va]), "a patient is in both groups"
            assert len(set(y[tr])) == 2 and len(set(y[va])) == 2, "a split lost one class"
            yield {"repeat": rep, "fold": k, "train": tr, "val": va}


def holdout(y, groups, seed=42, n_splits=4):
    """One fixed training and validation split (about one quarter of patients in validation)."""
    return next(patient_folds(y, groups, n_splits=n_splits, n_repeats=1, seed=seed))
