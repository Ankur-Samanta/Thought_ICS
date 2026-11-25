"""Appendix A tree: one real Thought-ICS correction trace (oracle-verified run), drawn as a tree.

    python scripts/paper/tree_example.py [out_pdf]

Example: Qwen-32B, MATH-500 #1172 (the 3-judge oracle places the first error at step 3).
Iteration 1 localizes step 4, so its prefix keeps step 3 and the flawed approach with it, and it
fails; iteration 2 localizes step 3 (clean prefix), changes approach and succeeds. Thought text is
shortened for display (marked with ...).
"""
import glob
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper_assets" / "tree_visualization_vertical.pdf"
sys.path.insert(0, str(REPO))
from thought_ics.benchmarks import load_benchmark, run_tag  # noqa: E402

MODEL, DS, PID, ORACLE_STEP = "qwen32b", "math500", "test/precalculus/1172.json", 3
TITLE = "Thought-ICS correction trace: Qwen-32B, MATH-500 (oracle verification)"

KEPT, KEPT_EDGE = "#d4ede2", "#127f5a"
ERR, ERR_EDGE = "#fbd9d9", "#e34948"
DISC, DISC_EDGE = "#eeeeec", "#b5b3aa"
Q, Q_EDGE = "#f4f4f2", "#52514e"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.usetex": False})


def plain(t, n=150):
    """LaTeX thought text -> short plain text for display."""
    t = re.sub(r"\\begin\{[pb]matrix\}(.*?)\\end\{[pb]matrix\}",
               lambda m: "(" + ", ".join(x.strip() for x in m.group(1).split(r"\\") if x.strip()) + ")", t, flags=re.S)
    t = re.sub(r"\\begin\{vmatrix\}(.*?)\\end\{vmatrix\}", "det(...)", t, flags=re.S)
    t = re.sub(r"\\(?:left|right|displaystyle|,|;|!)", "", t)
    t = re.sub(r"\\(?:vec|mathbf|mathrm|text|operatorname|boxed)\{([^{}]*)\}", r"\1", t)
    prev = None
    while prev != t:   # alternate until a fixpoint: \frac and \sqrt nest inside each other
        prev = t
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
        t = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", t)
    t = t.replace("\\cdot", "\u00b7").replace("\\times", "x").replace("\\cdots", "...").replace("\\tan", "tan")
    t = t.replace(r"\|", "|").replace(r"\[", "").replace(r"\]", "").replace(r"\(", "").replace(r"\)", "")
    t = re.sub(r"\^\{([^{}]*)\}", r"^\1", t)
    t = re.sub(r"_\{([^{}]*)\}", r"_\1", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t).replace("{", "").replace("}", "").replace("$", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " ..."


d = sorted(glob.glob(str(REPO / f"results/{MODEL}_{run_tag(DS)}_oracle_2*/")))[-1]
rec = next(json.loads(l) for l in open(Path(d) / "thought_results.jsonl") if json.loads(l)["problem_id"] == PID)
gens = [it for it in rec["iterations"] if "gen_status" in it]
c0, c1, c2 = (g["chain"] for g in gens[:3])
s1, s2 = gens[1]["localization"]["error_step"], gens[2]["localization"]["error_step"]
prob = next(p for p in load_benchmark(DS) if p["unique_id"] == PID)

W, LH, PAD, GAP = 4.2, 0.17, 0.12, 0.16   # box width, line height, inner padding, vertical gap
WRAP = 66
fig = plt.figure(figsize=(11, 15))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 11)
ax.set_ylim(-15, 0)
ax.axis("off")


def box(x, y, text, num=None, face=DISC, edge=DISC_EDGE, wrap=WRAP, bold=False, w=None):
    w = W if w is None else w
    lines = textwrap.wrap(text, wrap)
    h = len(lines) * LH + 2 * PAD
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=face, ec=edge, lw=1.1))
    tx = x + (0.32 if num is not None else 0.12)
    for i, ln in enumerate(lines):
        ax.text(tx, y - PAD - (i + 0.75) * LH, ln, fontsize=7.6, color=INK, fontweight="bold" if bold else "normal")
    if num is not None:
        ax.text(x + 0.12, y - h / 2, str(num), fontsize=9, color=edge, fontweight="bold", va="center")
    return y - h


def column(x, y, steps, start, colors, outcome=None, label=None, w=None, oracle=False, n=150):
    w = W if w is None else w
    wrap = int(WRAP * w / W)
    """steps: list of thought texts numbered from `start`; colors: per-step (face, edge)."""
    nchar = n
    for k, (t, (fc, ec)) in enumerate(zip(steps, colors)):
        n = start + k
        y = box(x, y, plain(t, nchar), num=n, face=fc, edge=ec, w=w, wrap=wrap)
        if oracle and n == ORACLE_STEP:
            ax.text(x + w + 0.08, y + 0.18, "oracle:\nfirst error", fontsize=7.5, color=INK2, va="bottom")
        y -= GAP * 0.6
    if outcome:
        ans, ok = outcome
        label_text = f"answer {plain(ans)}  {'✓' if ok else '✗'}"
        bw = max(1.8, 0.075 * len(label_text))          # badge grows with the answer text
        ax.add_patch(FancyBboxPatch((x + w / 2 - bw / 2, y - 0.32), bw, 0.3, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=KEPT_EDGE if ok else ERR_EDGE, ec="none"))
        ax.text(x + w / 2, y - 0.17, label_text, ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold")
        if label:
            ax.text(x + w / 2, y - 0.5, label, ha="center", va="top", fontsize=8.5, color=INK2)
        y -= 0.75
    return y


ax.text(5.5, -0.25, TITLE, ha="center", fontsize=12, color=INK, fontweight="bold")
QTEXT = (re.sub(r"\.([A-Z])", r". \1", plain(prob["problem"].split("Your final answer")[0], 400))
         + f'   (correct answer: {plain(prob["answer"])})')
y = box(3.4, -0.6, QTEXT, face=Q, edge=Q_EDGE, wrap=70)
y -= GAP
K = (KEPT, KEPT_EDGE)
y_fork = column(3.4, y, c0[:s2 - 1], 1, [K] * (s2 - 1), label="kept")   # the prefix kept throughout

# left: the steps that iterations 0 and 1 share beyond the kept prefix, then their two endings
xl, xr = 0.5, 6.3
ys = column(xl, y_fork - 0.2, c0[s2 - 1:s1 - 1], s2,
            [(ERR, ERR_EDGE)] + [(DISC, DISC_EDGE)] * (s1 - s2 - 1), oracle=True)   # localized in iteration 2
ax.text(xl, y_fork - 0.08, "iterations 0-1", fontsize=8.5, color=INK2)
# iteration 0 ending (steps 7.. of c0) and iteration 1 ending (steps 7.. of c1), side by side and narrower
WN = 2.55
ya = column(xl - 0.35, ys - 0.25, c0[s1 - 1:], s1, [(ERR, ERR_EDGE)] + [(DISC, DISC_EDGE)] * (len(c0) - s1),
            outcome=(gens[0]["answer"], False), label="iteration 0", w=WN, n=105)
yb = column(xl - 0.35 + WN + 0.15, ys - 0.25, c1[s1 - 1:], s1, [(DISC, DISC_EDGE)] * (len(c1) - s1 + 1),
            outcome=(gens[1]["answer"], False), label="iteration 1", w=WN, n=105)
# right: iteration 2 resamples from the step it localized and succeeds
ax.text(xr, y_fork - 0.08, "iteration 2", fontsize=8.5, color=INK2)
column(xr, y_fork - 0.2, c2[s2 - 1:], s2, [K] * (len(c2) - s2 + 1), outcome=(gens[2]["answer"], True))

# fork markers
for xx in (xl + W / 2, xr + W / 2):
    ax.annotate("", xy=(xx, y_fork - 0.2), xytext=(5.5, y_fork + 0.02),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=0, shrinkB=2))
for xx in (xl - 0.35 + WN / 2, xl - 0.35 + WN + 0.15 + WN / 2):
    ax.annotate("", xy=(xx, ys - 0.25), xytext=(xl + W / 2, ys + 0.02),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=0, shrinkB=2))

# legend
lx, ly = 8.6, -0.75
for i, (fc, ec, name) in enumerate(((KEPT, KEPT_EDGE, "Kept / correct"), (ERR, ERR_EDGE, "Localized error step"),
                                    (DISC, DISC_EDGE, "Discarded"))):
    ax.add_patch(FancyBboxPatch((lx, ly - 0.28 * i - 0.16), 0.3, 0.16, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=fc, ec=ec, lw=1))
    ax.text(lx + 0.4, ly - 0.28 * i - 0.08, name, fontsize=8, va="center", color=INK)

H = -min(ya, yb) + 0.5          # 1 data unit = 1 inch, so cropping keeps the text scale
ax.set_ylim(-H, 0)
fig.set_size_inches(11, H)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=160, bbox_inches="tight")
print("wrote", OUT, "| localized steps", s1, s2, "| answers", [g["answer"] for g in gens[:3]])
