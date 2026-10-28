# VariantPulse Quantum: A Four Gene Lung Cancer Panel with Quantum Optimisation

Team Kanban entry to the IBM Qiskit Fall Fest 2026, Quantum Biology Track, Khalifa University.

The task is to choose exactly four genes that separate lung adenocarcinoma tissue from normal lung tissue, using quantum optimisation in Qiskit, and to compare the result fairly with classical methods. Each gene is one binary variable, and the choice of a panel is written as a QUBO that rewards informative genes, penalises redundant pairs of genes, and penalises any panel that does not have exactly four genes. The QUBO is solved with QAOA in Qiskit and compared with an exact search on the same QUBO, LASSO, mRMR, random forest importance and simulated annealing.

## Rules we follow

- The panel is built from the team patient pool only. No outside data is used to choose genes.
- All samples from one patient stay together, in training or in validation.
- The shortlist of genes is chosen from training patients only, inside every split.
- The scoring classifier is the one supplied by the organisers and is never changed.
- Every seed and setting is written down, and library versions are pinned in `requirements.txt`.

## Data

The dataset and the scoring file belong to the organisers and are not included in this repository. To run the code, place the normalised matrix as `data/log_cpm_matrix.xlsx`, the metadata as `data/metadata.xlsx`, and the scoring file as `external/panel_checker.py`.

## Status

Work in progress for the midpoint submission on 28 October 2026.

## Team

Huda Mueen (team lead), Al Ain University, Data Science and AI.
Awaiz Ahmed, Al Ain University, Software Engineering.
Bilal Feroz Khan, Al Ain University, Software Engineering.
Kareena Mueenuddin, Amity University, Mechatronics Engineering.
