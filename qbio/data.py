"""Load the team pool: normalised log2(CPM + 1) expression and patient labels.

The organisers supply one value per gene per patient, so one patient is one sample.
Label 1 is lung adenocarcinoma (LUAD) and label 0 is normal tissue.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MATRIX_XLSX = DATA / "log_cpm_matrix.xlsx"
META_XLSX = DATA / "metadata.xlsx"
EXPR_CSV = DATA / "pool_expression.csv"
LABELS_CSV = DATA / "pool_labels.csv"


def prepare():
    """Convert the organiser Excel files once into plain CSV (patients x genes)."""
    genes_by_patient = pd.read_excel(MATRIX_XLSX, index_col=0)
    meta = pd.read_excel(META_XLSX).set_index("patient_id")
    assert not genes_by_patient.index.duplicated().any(), "duplicate gene names"
    assert set(genes_by_patient.columns) == set(meta.index), "patients differ between files"

    order = sorted(genes_by_patient.columns, key=lambda p: int(p[1:]))
    expr = genes_by_patient[order].T
    expr.index.name = "patient"
    meta = meta.loc[order]

    labels = pd.DataFrame(
        {
            "patient": order,
            "label": (meta["histology"] == "LUAD").astype(int).to_numpy(),
            "female": (meta["sex"] == "female").astype(int).to_numpy(),
            "age": meta["age"].to_numpy(),
            "histology": meta["histology"].to_numpy(),
        }
    ).set_index("patient")
    expr.to_csv(EXPR_CSV)
    labels.to_csv(LABELS_CSV)
    return expr, labels


def load_pool():
    """Return (X, meta). X is patients x genes, meta has label, female, age, histology."""
    if not (EXPR_CSV.exists() and LABELS_CSV.exists()):
        return prepare()
    X = pd.read_csv(EXPR_CSV, index_col=0)
    meta = pd.read_csv(LABELS_CSV, index_col=0)
    assert (X.index == meta.index).all(), "expression and labels do not line up"
    return X, meta
