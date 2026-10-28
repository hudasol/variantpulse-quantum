"""Convert the organiser Excel files into CSV and print a short data summary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbio.data import prepare

X, meta = prepare()
print("patients x genes:", X.shape)
print("tumour / normal:", int(meta["label"].sum()), "/", int((meta["label"] == 0).sum()))
print("female patients:", int(meta["female"].sum()), "of", len(meta))
print("genes that are zero in every patient:", int((X.sum(axis=0) == 0).sum()))
