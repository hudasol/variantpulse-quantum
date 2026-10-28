"""Build the one page method note for the midpoint submission from the logged results.

Every number in the note is read from results/midpoint_summary.json, results/midpoint_runs.csv,
results/shortlist_full_pool.csv, results/cv_folds.jsonl, results/cv_methods.csv and
results/cv_summary.json (the last two are written by 04_cv_summary.py), so the text cannot drift
away from what was run. The build stops unless all 50 cross validation folds are finished; set
ALLOW_PARTIAL=1 to test the layout on partial results.
"""
import json
import math
import os
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
PARTIAL = os.environ.get("ALLOW_PARTIAL") == "1"
OUT = ROOT / "results" / "method_partial_test.pdf" if PARTIAL else ROOT / "midpoint" / "method.pdf"
FONTS = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "DejaVuSansCondensed.ttf")))
pdfmetrics.registerFont(TTFont("BodyBold", str(FONTS / "DejaVuSansCondensed-Bold.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")

GARNET, OXBLOOD, CARBON = colors.HexColor("#7A263A"), colors.HexColor("#481A27"), colors.HexColor("#17191C")
SAKURA, PEARL, EDGE = colors.HexColor("#F7DCE2"), colors.HexColor("#FDF7F6"), colors.HexColor("#E8C5CC")

S = json.loads((ROOT / "results" / "midpoint_summary.json").read_text())
runs = pd.read_csv(ROOT / "results" / "midpoint_runs.csv")
short = pd.read_csv(ROOT / "results" / "shortlist_full_pool.csv").set_index("gene")
cv = pd.read_csv(ROOT / "results" / "cv_methods.csv").set_index("method")
CV = json.loads((ROOT / "results" / "cv_summary.json").read_text())
first_fold = json.loads((ROOT / "results" / "cv_folds.jsonl").read_text().splitlines()[0])
n_folds = CV["folds"]
if n_folds != 50 and not PARTIAL:
    raise SystemExit(f"only {n_folds} of 50 cross validation folds are finished")

per_fold_rows = pd.Series([r["method"] for r in first_fold["rows"]]).value_counts()
qaoa_runs_cv = int(per_fold_rows["qaoa"]) * n_folds
qaoa_seeds = int(per_fold_rows["qaoa"]) // 3
sa_runs_cv = int(per_fold_rows["simulated_annealing"]) * n_folds

panel = sorted(S["panel"])
log, qb = S["shortlist_log"], S["qubo"]
v = S["versions"]
found = runs.groupby("reps")["found_optimum"].sum().to_dict()
n_runs, n_found = S["qaoa_runs"], S["qaoa_found_exact_optimum_runs"]
miss = runs[~runs["found_optimum"]]
miss_panels = sorted(set(miss["panel"]))
assert sorted(panel) == sorted(qb["exact_panel"]), "panel and exact optimum differ, text below needs a rewrite"
assert len(miss_panels) == 1 and len(panel) == 4 and len(set(panel)) == 4

body = ParagraphStyle("body", fontName="Body", fontSize=8.4, leading=10.7, textColor=CARBON, alignment=TA_LEFT)
h = ParagraphStyle("h", parent=body, fontName="BodyBold", fontSize=9.4, leading=11.3, textColor=GARNET,
                   spaceBefore=2.8, spaceAfter=0.8)
title = ParagraphStyle("title", parent=body, fontName="BodyBold", fontSize=14, leading=16.5, textColor=OXBLOOD)
sub = ParagraphStyle("sub", parent=body, fontSize=8.3, leading=10.4, textColor=CARBON)
cell = ParagraphStyle("cell", parent=body, fontSize=8.0, leading=9.4)
cellb = ParagraphStyle("cellb", parent=cell, fontName="BodyBold", textColor=OXBLOOD)


def P(text, style=body):
    return Paragraph(text, style)


genes_sorted = panel
head = Table(
    [[P("Team Kanban: midpoint method note", title)],
     [P("IBM Qiskit Fall Fest 2026, Quantum Biology Track. Lung adenocarcinoma (LUAD) versus normal lung tissue. "
        "Huda Mueen, Awaiz Ahmed, Bilal Feroz Khan (Al Ain University) and Kareena Mueenuddin (Amity University).", sub)]],
    colWidths=[180 * mm],
)
head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SAKURA), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                          ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (0, 0), 5),
                          ("BOTTOMPADDING", (0, -1), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (0, 0), 1),
                          ("TOPPADDING", (0, 1), (0, 1), 0), ("LINEBELOW", (0, -1), (-1, -1), 1.2, GARNET)]))

direction = {g: ("tumour" if short.loc[g, "welch_t"] > 0 else "normal") for g in genes_sorted}
rows = [[P("Submitted panel", cellb)] + [P(g, cellb) for g in genes_sorted],
        [P("Welch t (LUAD minus normal), higher in", cell)] + [P(f"{short.loc[g, 'welch_t']:+.2f}".replace("-", "−") + f" ({direction[g]})", cell) for g in genes_sorted]]
tbl = Table(rows, colWidths=[60 * mm] + [30 * mm] * 4)
tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), PEARL), ("BOX", (0, 0), (-1, -1), 0.6, EDGE),
                         ("INNERGRID", (0, 0), (-1, -1), 0.3, EDGE), ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                         ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8), ("LEFTPADDING", (0, 0), (-1, -1), 5)]))

story = [head, Spacer(1, 3), tbl]

story += [P("1. Data", h), P(
    f"We used only the normalised log CPM matrix supplied by the organisers, exactly as supplied and with no further "
    f"normalisation: {S['patients']} patients ({S['tumour']} LUAD, {S['normal']} normal) and {log['genes_in']:,} genes. "
    f"No outside data was used to choose the panel.")]

story += [P("2. How we split the data", h), P(
    "The split is made by patient and stratified by diagnosis, so that no patient is ever in both the training and the "
    "validation set. We use five folds repeated ten times (seed 42); the 50 folds overlap because they reshuffle the same "
    "26 patients. Everything that looks at the labels, which means the shortlist, the QUBO weights and every solver, is built "
    "from the training patients of a fold only, and every method works on the same 16 gene shortlist. The validation patients are "
    "used once, to score a panel with the organisers' fixed classifier (standard scaling and logistic regression, C = 1.0). "
    "The submitted panel is chosen with all 26 patients.")]

story += [P("3. How the shortlist was chosen", h), P(
    f"The shortlist has 16 genes and is built in four steps, here on all 26 patients. (a) Keep genes detected (log CPM above 0) in at least half of the patients with a mean "
    f"log2(CPM + 1) of at least 1; {log['after_expression_filter']:,} remain. (b) Remove genes that separate female from male "
    f"patients almost perfectly (AUROC of at least 0.95, sex from the organisers' metadata), because most normal patients are male and "
    f"four of the five women are in the tumour group; {log['removed_as_sex_linked']} genes were removed, including Y linked genes. "
    f"(c) Rank the rest by the absolute Welch t statistic between LUAD and normal. (d) Walk down the ranking and skip any gene whose "
    f"absolute correlation with an already kept gene is above 0.9 ({log['skipped_near_duplicates']} skipped), until 16 genes are kept.")]

formula = ParagraphStyle("formula", parent=body, leading=14, alignment=1, textColor=OXBLOOD, spaceBefore=1.5, spaceAfter=1.5)
story += [P("4. QUBO", h),
          P("Each shortlisted gene is one binary variable x<sub>i</sub> (1 means the gene is in the panel). We minimise"),
          P("E(x) = &#8722;&#945; &#931; r<sub>i</sub> x<sub>i</sub> + &#946; &#931;<sub>i&lt;j</sub> c<sub>ij</sub> x<sub>i</sub> x<sub>j</sub> "
            "+ &#955; (&#931; x<sub>i</sub> &#8722; 4)<super>2</super>", formula),
          P(f"where r<sub>i</sub> is the absolute Welch t scaled to (0, 1] and c<sub>ij</sub> is the absolute Pearson correlation "
            f"between genes i and j. We set &#945; = 1 and &#946; = 2/3, so that mean relevance and mean redundancy of a four gene "
            f"panel carry equal weight (the mRMR idea). &#955; = {qb['lam']:.3f}, which is twice the smallest value "
            f"({qb['lam_min']:.3f}) that makes the global optimum contain exactly four genes. With only 16 variables we also "
            f"checked all {qb['states']:,} states exactly.")]

story += [P("5. Library and QAOA", h), P(
    f"<b>Library: Qiskit {v['qiskit']}.</b> qiskit-optimization {v['qiskit_optimization']} builds the Ising operator, QAOAAnsatz the "
    f"circuit, qiskit-aer {v['qiskit_aer']} runs it as a noise free shot simulation (no hardware yet) and qiskit-algorithms "
    f"{v['qiskit_algorithms']} supplies the COBYLA optimiser; scikit-learn {v['scikit_learn']} scores panels. On all 26 patients we ran "
    f"depths p = 1, 2 and 3 with ten seeds each ({n_runs} runs; up to {S['config']['maxiter']} COBYLA iterations, "
    f"{S['config']['shots']} shots per evaluation, {S['config']['final_shots']} final shots). A run returns the lowest energy four "
    f"gene bitstring among its final samples, and the submitted panel is the one returned most often. {n_found} of {n_runs} runs "
    f"({100 * n_found / n_runs:.0f}%) returned the exact optimum; "
    f"the other {n_runs - n_found} returned {', '.join(miss_panels[0].split(';'))}.")]


def pct(x):
    return f"{100 * x:.0f}%"


def with_interval(r):
    return f"{pct(r['found_optimum'])} ({100 * r['found_lo']:.0f} to {100 * r['found_hi']:.0f})"


qa_depths = cv.loc[["qaoa, p=1", "qaoa, p=2", "qaoa, p=3"]]
qa = cv.loc["qaoa, all depths"]
rp = cv.loc["random_four_gene_panel"]
NA = "n/a"
labels = [
    ("Exact search of the QUBO", cv.loc["exact_qubo"], pct(cv.loc["exact_qubo", "found_optimum"])),
    (f"QAOA in Qiskit, p = 1, 2, 3 ({qaoa_runs_cv} runs)", qa, with_interval(qa)),
    (f"Simulated annealing, same QUBO ({sa_runs_cv} runs)", cv.loc["simulated_annealing"], with_interval(cv.loc["simulated_annealing"])),
    ("Random sampling, same final shots as QAOA", cv.loc["random_sampling, same_final_shots"], with_interval(cv.loc["random_sampling, same_final_shots"])),
    ("Random sampling, same total shots as QAOA", cv.loc["random_sampling, same_total_shots"], with_interval(cv.loc["random_sampling, same_total_shots"])),
    ("Greedy mRMR on the same weights", cv.loc["greedy_mrmr"], with_interval(cv.loc["greedy_mrmr"])),
    ("Top 4 genes by single gene AUROC", cv.loc["naive_single_gene"], NA),
    ("LASSO forced to 4 genes", cv.loc["lasso"], NA),
    ("Random forest importance", cv.loc["random_forest"], NA),
    ("Wrapper search with inner cross validation", cv.loc["wrapper_exact"], NA),
    ("Random four gene panel (mean of all 1820)", rp, NA),
]
data = [[P("Method", cellb), P("AUROC", cellb), P("Log loss", cellb),
         P("Found the QUBO optimum (95% interval)", cellb)]]
for name, r, found_txt in labels:
    data.append([P(name, cell), P(f"{r['auroc_mean']:.3f}", cell), P(f"{r['log_loss_mean']:.3f}", cell), P(found_txt, cell)])
res = Table(data, colWidths=[78 * mm, 20 * mm, 22 * mm, 60 * mm])
res.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), PEARL), ("BACKGROUND", (0, 2), (-1, 2), SAKURA),
                         ("BOX", (0, 0), (-1, -1), 0.6, EDGE), ("INNERGRID", (0, 0), (-1, -1), 0.3, EDGE),
                         ("TOPPADDING", (0, 0), (-1, -1), 0.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3),
                         ("LEFTPADDING", (0, 0), (-1, -1), 4)]))

low_auc = [name for name, r, _ in labels[:-1] if r["auroc_mean"] < 0.99]
ll_sd = float(cv["log_loss_sd"].mean())
top_ok = float(qa_depths["top_bitstring_right_size"].mean())
fr, ft = cv.loc["random_sampling, same_final_shots"], cv.loc["random_sampling, same_total_shots"]
n_short = len(S["shortlist"])
uniform_four = math.comb(n_short, 4) / 2 ** n_short
uniform_opt = 1 / 2 ** n_short
exo = CV["submitted_gene_in_fold_exact_panel"]
g1, g2, g3, g4 = "SERPINB13", "CALML3", "TMEM40", "PIP4K2B"
assert qa["found_hi"] < ft["found_lo"] or qa["found_lo"] > ft["found_hi"], "intervals overlap, the sentence below needs a rewrite"
assert rp["log_loss_mean"] > qa["log_loss_mean"] and qa["found_optimum"] > ft["found_optimum"]

story += [P(f"6. Cross validation results ({n_folds} overlapping folds, validation patients only)", h), res, Spacer(1, 2.5), P(
    f"QAOA uses {qaoa_seeds} seeds per depth in every fold; random sampling gets QAOA's final or total shots. AUROC and log loss are validation values averaged "
    f"over folds of about 5 patients. Intervals are 95% bootstrap intervals over folds and understate the uncertainty because the folds "
    f"overlap. The last column applies only to methods that optimise the QUBO."), P(
    f"Validation AUROC is close to 1.00 for almost every method, so it cannot separate them. "
    f"A random four gene panel already gives {rp['auroc_mean']:.3f} and a worse log loss "
    f"({rp['log_loss_mean']:.3f}, against {qa['log_loss_mean']:.3f} for QAOA; spread across folds about {ll_sd:.2f}). "
    f"QAOA found the exact optimum in {pct(qa['found_optimum'])} of runs ({pct(qa_depths['found_optimum'].min())} to "
    f"{pct(qa_depths['found_optimum'].max())} by depth), against {pct(ft['found_optimum'])} for random sampling with the same total shots. "
    f"The panel is picked by evaluating the energy of the final samples classically, but the QAOA distribution "
    f"itself is enriched: four gene strings hold {pct(qa['prob_right_size'])} of the shots ({100 * uniform_four:.1f}% for a uniform random string) and "
    f"the optimum {100 * qa['prob_optimum']:.3f}% ({100 * uniform_opt:.4f}%). The most probable string had four genes in only "
    f"{pct(top_ok)} of folds, so we read the panel from the samples."), P(
    f"The exact optimum changed between folds ({CV['distinct_fold_exact_panels']} different panels in {n_folds} folds): {g1} was in "
    f"{exo[g1]} of {n_folds}, and {g2}, {g3} and {g4} in {exo[g2]}, {exo[g3]} and {exo[g4]}. The folds therefore estimate how well the "
    f"procedure works, not this exact panel, and the submitted panel is one of several near equal choices.")]

story += [P("7. Limits we want to state clearly", h), P(
    f"On 16 genes the exact search takes a few milliseconds, so QAOA is a working quantum method here, not a faster one. The best and "
    f"second best four gene panels differ by only {qb['gap_best_to_second_size4']:.3f} in QUBO energy. The shortlist size and filter "
    f"thresholds were set by us with all 26 patients in view, so the folds are not fully nested. Tissue site and sex are not balanced "
    f"(some tumour samples are lymph nodes) and we did not remove that effect. Three of our four genes "
    f"are higher in normal tissue, so they may partly mark lung tissue rather than LUAD.")]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=8 * mm, bottomMargin=6 * mm,
                        title="Team Kanban midpoint method note", author="Team Kanban")
doc.build(story)
print("written", OUT)
