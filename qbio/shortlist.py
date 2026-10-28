"""Choose a small shortlist of genes from TRAINING patients only.

Steps, all computed on the training patients passed in:
1. Expression filter: keep genes detected in at least half of the patients with a mean
   log2(CPM + 1) of at least 1, so very low counts do not drive the ranking.
2. Sex confound filter: in this pool most normal patients are male while several tumour
   patients are female, so genes that separate the sexes almost perfectly (for example XIST
   and Y linked genes) could look like cancer markers. They are removed using the training
   patients' own sex labels.
3. Rank the remaining genes by the absolute Welch t statistic between tumour and normal.
4. Walk down the ranking and skip genes that are nearly identical (absolute correlation above
   max_corr) to a gene already kept, until the shortlist has k genes.
"""
import numpy as np
from scipy import stats


def auroc_columns(V, positive):
    """AUROC of every column of V for the boolean mask `positive`, computed from ranks."""
    ranks = stats.rankdata(V, axis=0)
    n1 = int(positive.sum())
    n0 = int((~positive).sum())
    u = ranks[positive].sum(axis=0) - n1 * (n1 + 1) / 2.0
    return u / (n1 * n0)


def welch_t(V, y):
    """Welch t statistic (tumour minus normal) for every column of V."""
    t, _ = stats.ttest_ind(V[y == 1], V[y == 0], equal_var=False, axis=0)
    return np.nan_to_num(t)


def make_shortlist(
    X,
    y,
    female,
    k=16,
    min_detect=0.5,
    min_mean=1.0,
    sex_auc=0.95,
    max_corr=0.9,
    min_sex_group=3,
    pool=80,
):
    """Return a dict with the shortlist gene names, their t statistics and a filter log."""
    y = np.asarray(y)
    female = np.asarray(female).astype(bool)
    V = X.to_numpy(dtype=float)
    genes = np.asarray(X.columns)
    log = {"genes_in": int(V.shape[1])}

    expressed = ((V > 0).mean(axis=0) >= min_detect) & (V.mean(axis=0) >= min_mean)
    log["after_expression_filter"] = int(expressed.sum())

    sex_flag = np.zeros(V.shape[1], dtype=bool)
    if female.sum() >= min_sex_group and (~female).sum() >= min_sex_group:
        a = auroc_columns(V, female)
        sex_flag = (a >= sex_auc) | (a <= 1 - sex_auc)
    removed_sex = expressed & sex_flag
    log["removed_as_sex_linked"] = int(removed_sex.sum())
    log["sex_linked_genes"] = genes[removed_sex].tolist()

    keep = np.flatnonzero(expressed & ~sex_flag)
    t = welch_t(V[:, keep], y)
    top = keep[np.argsort(-np.abs(t))[:pool]]
    t_top = dict(zip(top, welch_t(V[:, top], y)))

    corr = np.abs(np.corrcoef(V[:, top], rowvar=False))
    chosen = []
    for pos, col in enumerate(top):
        if all(corr[pos, other] < max_corr for other in chosen):
            chosen.append(pos)
        if len(chosen) == k:
            break
    cols = top[chosen]
    log["skipped_near_duplicates"] = int(pos + 1 - len(chosen))
    return {
        "genes": genes[cols].tolist(),
        "columns": cols,
        "t": np.array([t_top[c] for c in cols]),
        "log": log,
    }
