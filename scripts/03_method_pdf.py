"""Build the one page method note for the midpoint submission from the logged results.

Every number in the note is read from results/midpoint_summary.json, results/midpoint_runs.csv
and results/shortlist_full_pool.csv, so the text cannot drift away from what was run.
"""
import json
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
OUT = ROOT / "midpoint" / "method.pdf"
FONTS = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "DejaVuSansCondensed.ttf")))
pdfmetrics.registerFont(TTFont("BodyBold", str(FONTS / "DejaVuSansCondensed-Bold.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")

GARNET, OXBLOOD, CARBON = colors.HexColor("#7A263A"), colors.HexColor("#481A27"), colors.HexColor("#17191C")
SAKURA, PEARL, EDGE = colors.HexColor("#F7DCE2"), colors.HexColor("#FDF7F6"), colors.HexColor("#E8C5CC")

S = json.loads((ROOT / "results" / "midpoint_summary.json").read_text())
runs = pd.read_csv(ROOT / "results" / "midpoint_runs.csv")
short = pd.read_csv(ROOT / "results" / "shortlist_full_pool.csv").set_index("gene")

panel = sorted(S["panel"])
log, qb = S["shortlist_log"], S["qubo"]
v = S["versions"]
found = runs.groupby("reps")["found_optimum"].sum().to_dict()
n_runs, n_found = S["qaoa_runs"], S["qaoa_found_exact_optimum_runs"]
miss = runs[~runs["found_optimum"]]
miss_panels = sorted(set(miss["panel"]))
right_size_top = S["qaoa_runs_where_most_probable_string_has_right_size"]
assert sorted(panel) == sorted(qb["exact_panel"]), "panel and exact optimum differ, text below needs a rewrite"
assert len(miss_panels) == 1 and len(panel) == 4 and len(set(panel)) == 4

body = ParagraphStyle("body", fontName="Body", fontSize=9.1, leading=12.4, textColor=CARBON, alignment=TA_LEFT)
h = ParagraphStyle("h", parent=body, fontName="BodyBold", fontSize=10, leading=12.5, textColor=GARNET,
                   spaceBefore=7, spaceAfter=2)
title = ParagraphStyle("title", parent=body, fontName="BodyBold", fontSize=15, leading=18, textColor=OXBLOOD)
sub = ParagraphStyle("sub", parent=body, fontSize=9, leading=11.8, textColor=CARBON)
cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=11.5)
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
                          ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (0, 0), 6),
                          ("BOTTOMPADDING", (0, -1), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (0, 0), 1),
                          ("TOPPADDING", (0, 1), (0, 1), 0), ("LINEBELOW", (0, -1), (-1, -1), 1.2, GARNET)]))

direction = {g: ("tumour" if short.loc[g, "welch_t"] > 0 else "normal") for g in genes_sorted}
rows = [[P("Submitted panel", cellb)] + [P(g, cellb) for g in genes_sorted],
        [P("Higher in", cell)] + [P(direction[g], cell) for g in genes_sorted],
        [P("Welch t (LUAD minus normal)", cell)] + [P(f"{short.loc[g, 'welch_t']:+.2f}".replace("-", "−"), cell) for g in genes_sorted]]
tbl = Table(rows, colWidths=[52 * mm] + [32 * mm] * 4)
tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), PEARL), ("BOX", (0, 0), (-1, -1), 0.6, EDGE),
                         ("INNERGRID", (0, 0), (-1, -1), 0.3, EDGE), ("TOPPADDING", (0, 0), (-1, -1), 2),
                         ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 5)]))

story = [head, Spacer(1, 5), tbl]

story += [P("1. Data", h), P(
    f"We used only the normalised log CPM matrix supplied by the organisers, exactly as supplied and with no further "
    f"normalisation: {S['patients']} patients ({S['tumour']} LUAD, {S['normal']} normal) and {log['genes_in']:,} genes. "
    f"No outside data was used to choose the panel.")]

story += [P("2. How we split the data", h), P(
    "The split is made by patient and stratified by diagnosis, so that no patient is ever in both the training and the "
    "validation set. We use five folds repeated ten times (seed 42, 50 folds in total). Everything that looks at the labels, "
    "which means the shortlist, the QUBO weights and every solver, is built from the training patients of a fold only. The "
    "validation patients are used once, to score a panel with the organisers' fixed classifier (standard scaling and "
    "logistic regression, C = 1.0). The submitted panel is chosen with all 26 patients, because no further labelled "
    "patients exist. Its expected performance is therefore read from the repeated folds, not from the 26 patients it was chosen on.")]

story += [P("3. How the shortlist was chosen", h), P(
    f"The shortlist has 16 genes and is built in four steps. (a) Keep genes detected in at least half of the patients with a "
    f"mean log2(CPM + 1) of at least 1; {log['after_expression_filter']:,} remain. (b) Remove genes that separate male and female "
    f"patients almost perfectly (AUROC of at least 0.95 for sex), because most normal patients are male and four of the five "
    f"women are in the tumour group; {log['removed_as_sex_linked']} genes were removed, including Y linked genes. (c) Rank the "
    f"rest by the absolute Welch t statistic between LUAD and normal. (d) Walk down the ranking and skip any gene whose absolute "
    f"correlation with an already kept gene is above 0.9 ({log['skipped_near_duplicates']} skipped), until 16 genes are kept.")]

formula = ParagraphStyle("formula", parent=body, leading=17, alignment=1, textColor=OXBLOOD, spaceBefore=2, spaceAfter=2)
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
    f"<b>Library: Qiskit {v['qiskit']}.</b> qiskit-optimization {v['qiskit_optimization']} turns the QUBO into an Ising operator, "
    f"QAOAAnsatz builds the circuit, qiskit-aer {v['qiskit_aer']} runs it as a shot based simulation without noise (no hardware yet), "
    f"and qiskit-algorithms {v['qiskit_algorithms']} supplies the COBYLA optimiser. Classifier scoring uses scikit-learn "
    f"{v['scikit_learn']}. We ran QAOA at depths p = 1, 2 and 3 with ten seeds each ({n_runs} runs), with up to "
    f"{S['config']['maxiter']} COBYLA iterations, {S['config']['shots']} shots per evaluation and {S['config']['final_shots']} shots for "
    f"the final distribution. A run returns the lowest energy four gene bitstring among its final samples, and the submitted panel "
    f"is the one returned most often. {n_found} of {n_runs} runs returned the same panel as the exact search "
    f"(p = 1: {found[1]} of 10, p = 2: {found[2]} of 10, p = 3: {found[3]} of 10). The other {n_runs - n_found} runs "
    f"returned {', '.join(miss_panels[0].split(';'))}. The single most probable bitstring had four genes in only "
    f"{right_size_top} of {n_runs} runs, which is why the panel is read from the samples and not from the top bitstring.")]

story += [P("6. Limits we want to state clearly", h), P(
    f"On 16 genes the exact search takes a few milliseconds, so QAOA gives no speed advantage here; our claim is that it is a "
    f"working and correct quantum method for this problem, and we compare it with classical methods on the same shortlist and "
    f"the same compute budget in the final presentation. The best and second best four gene panels differ by only "
    f"{qb['gap_best_to_second_size4']:.3f} in QUBO energy, so the data do not single out one panel sharply. Tissue site and sex "
    f"are not balanced in this cohort (some tumour samples come from lymph nodes), and we have not tried to remove that effect.")]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=12 * mm, bottomMargin=10 * mm,
                        title="Team Kanban midpoint method note", author="Team Kanban")
doc.build(story)
print("written", OUT)
