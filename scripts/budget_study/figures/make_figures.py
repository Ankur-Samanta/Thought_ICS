"""Regenerate the paper's oracle-verified figures (everything before the autonomous section) from
the budget-study runs, with one validated palette.

    python scripts/budget_study/figures/make_figures.py [out_dir]

Default out_dir: TREE/thoughtics_paper/paper_v2/figures (the copy of the Meta version; the ICML and
original Meta versions are not touched). Writes, at the original sizes:
  l2_errorbar                       self-correction lift over the shared first attempt (oracle-verified)
  self_localization_vs_oracle       Thought-ICS first localization vs the 3-judge oracle
  self_localization_vs_oracle_token Token-ICS, same items
  edit_accuracy_clean_vs_erroneous  fix rate of the first resample, clean vs erroneous prefix (Thought-ICS)
  localization_error_distribution   self - oracle deviation (Thought-ICS), one panel per model family

Data: experiments_budget_study/full/*_L2_shared_fbattempt_* (math datasets regraded with
math_verify, runs truncated at the first correct attempt, as in report.py) and the oracle rows in
experiments_budget_study/oracle_{full,slm}/rows.json (scripts/budget_study/oracle/analyze.py).
Palette checked with the dataviz validator (methods: blue/orange; prefixes: exact/earlier/erroneous;
families: one ordinal ramp each).
"""
import glob
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from math_verify import parse, verify  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments_budget_study"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/asamanta1/SIERA/TREE/thoughtics_paper/paper_v2/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (validated) & chrome -------------------------------------------------------
THOUGHT, TOKEN = "#2a78d6", "#eb6834"
EXACT, EARLIER, ERRONEOUS, FAILED = "#127f5a", "#52c79b", "#e34948", "#b5b3aa"
FAMILY = {"llama": ["#f0955f", "#d8571f", "#93350f"], "qwen": ["#6da7ec", "#2a78d6", "#184f95"],
          "gptoss": ["#8f83ea", "#4a3aa7"]}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "axes.labelsize": 14, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 11, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
ORDER = ["llama3b", "qwen7b", "llama8b", "qwen14b", "gptoss20b", "qwen32b", "llama70b", "gptoss120b"]
SHORT = {"llama3b": "L3B", "qwen7b": "Q7B", "llama8b": "L8B", "qwen14b": "Q14B", "gptoss20b": "G20B",
         "qwen32b": "Q32B", "llama70b": "L70B", "gptoss120b": "G120B"}
LONG = {"llama3b": "Llama-3B", "llama8b": "Llama-8B", "llama70b": "Llama-70B", "qwen7b": "Qwen-7B",
        "qwen14b": "Qwen-14B", "qwen32b": "Qwen-32B", "gptoss20b": "gpt-oss-20B", "gptoss120b": "gpt-oss-120B"}
FAMILIES = [("Llama", "llama", ["llama3b", "llama8b", "llama70b"]), ("Qwen", "qwen", ["qwen7b", "qwen14b", "qwen32b"]),
            ("gpt-oss", "gptoss", ["gptoss20b", "gptoss120b"])]
MATH = {"amc23", "aime", "math500"}
_mv = {}


def ok(ans, gt, ds, strict):
    if strict or ds not in MATH or ans is None:
        return bool(strict)
    if (ans, gt) not in _mv:
        try:
            A, G = parse("$" + ans + "$"), parse("$" + gt + "$")
            _mv[(ans, gt)] = bool(A and G and verify(G, A))
        except Exception:  # noqa: BLE001
            _mv[(ans, gt)] = False
    return _mv[(ans, gt)]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ---- Fig: l2_errorbar --------------------------------------------------------------------
def first_correct(rec, ds):
    """(init_ok, final_ok) under math_verify, run truncated at the first correct attempt."""
    for k, it in enumerate(rec["iterations"]):
        if "gen_status" in it and ok(it["answer"], rec["ground_truth"], ds, it["correct"]):
            return k == 0, True
    return False, False


rows_l2 = defaultdict(list)
for d in sorted(glob.glob(str(EXP / "full" / "*_L2_shared_fbattempt_*/"))):
    d = Path(d)
    if not (d / "token_results.jsonl").exists():
        continue
    c = json.load(open(d / "config.json"))
    load = lambda m: {json.loads(l)["problem_id"]: json.loads(l) for l in open(d / f"{m}_results.jsonl")}
    T, K = load("thought"), load("token")
    for pid, t in T.items():
        i0, ft = first_correct(t, c["dataset"])
        _, fk = first_correct(K[pid], c["dataset"])
        rows_l2[c["model"]].append((i0, ft, fk))


def boot_lift(rows, j, B=5000, seed=0):
    rnd = random.Random(seed)
    f = lambda s: 100 * sum(r[j] - r[0] for r in s) / len(s)
    v = sorted(f([rows[rnd.randrange(len(rows))] for _ in rows]) for _ in range(B))
    return f(rows), v[int(.025 * B)], v[int(.975 * B)]


fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(ORDER))
w = 0.36
print("\nl2_errorbar: lift over the shared first attempt [95% bootstrap CI]")
for off, j, name, col in ((-w / 2, 2, "Token-ICS", TOKEN), (w / 2, 1, "Thought-ICS", THOUGHT)):
    est = [boot_lift(rows_l2[m], j) for m in ORDER]
    mid = [e[0] for e in est]
    ax.bar(x + off, mid, w * 0.94, color=col, label=name, zorder=2)
    ax.errorbar(x + off, mid, yerr=[[e[0] - e[1] for e in est], [e[2] - e[0] for e in est]],
                fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=3)
    for m, e in zip(ORDER, est):
        print(f"  {name:11s} {m:10s} {e[0]:5.1f} [{e[1]:5.1f}, {e[2]:5.1f}]  (n={len(rows_l2[m])})")
ax.set_xticks(x, [SHORT[m] for m in ORDER])
ax.set_xlabel("Model")
ax.set_ylabel("Self-correction lift (%)")
ax.set_ylim(0, 40)
ax.legend(loc="upper right", title="Under oracle verification", title_fontsize=12, alignment="left")
save(fig, "l2_errorbar")

# ---- oracle rows ------------------------------------------------------------------------
orows = [r for o in ("oracle_full", "oracle_slm") for r in json.load(open(EXP / o / "rows.json"))]
pair = defaultdict(dict)
for r in orows:
    pair[r["id"]][r["method"]] = r
paired = [v for v in pair.values() if "thought" in v and "token" in v]


def loc_fig(method, name, legend_loc="upper left"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(ORDER))
    w = 0.38
    print(f"\n{name}: first localization vs oracle ({method}), all localizations")
    for i, m in enumerate(ORDER):
        R = [v[method] for v in paired if v[method]["model"] == m]
        n = len(R)
        p = {k: 100 * sum(r["cat"] == k for r in R) / n for k in ("exact", "earlier", "later", "failed")}
        clean = p["exact"] + p["earlier"]
        bad = p["later"] + p["failed"]
        ax.bar(i - w / 2, p["exact"], w * 0.94, color=EXACT, zorder=2)
        ax.bar(i - w / 2, p["earlier"], w * 0.94, bottom=p["exact"], color=EARLIER, edgecolor=SURFACE, linewidth=1, zorder=2)
        ax.bar(i + w / 2, p["later"], w * 0.94, color=ERRONEOUS, zorder=2)
        ax.bar(i + w / 2, p["failed"], w * 0.94, bottom=p["later"], color=FAILED, edgecolor=SURFACE, linewidth=1,
               hatch="////", zorder=2)
        clo, chi = wilson(round(clean * n / 100), n)
        ax.errorbar(i - w / 2, clean, yerr=[[clean - clo], [chi - clean]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        lo, hi = wilson(round(bad * n / 100), n)
        ax.errorbar(i + w / 2, bad, yerr=[[bad - lo], [hi - bad]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        ax.text(i - w / 2, chi + 1.5, f"{clean:.0f}%", ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold")
        print(f"  {m:10s} n={n:3d} exact {p['exact']:4.0f} earlier {p['earlier']:4.0f} | later {p['later']:4.0f} failed {p['failed']:4.0f} | clean {clean:4.0f}")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=EXACT, label="Clean: exact match"), Patch(color=EARLIER, label="Clean: earlier"),
                       Patch(color=ERRONEOUS, label="Erroneous: later"),
                       Patch(facecolor=FAILED, hatch="////", edgecolor=SURFACE, label="Failed localization")],
              loc=legend_loc, fontsize=9, ncol=2, handlelength=1.4, columnspacing=1.0)
    ax.set_xticks(x, [SHORT[m] for m in ORDER], fontsize=11)
    ax.set_xlabel("Model")
    ax.set_ylabel("First localizations (%)")
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    save(fig, name)


loc_fig("thought", "self_localization_vs_oracle")
loc_fig("token", "self_localization_vs_oracle_token")

# ---- Fig: edit_accuracy_clean_vs_erroneous (Thought-ICS) ---------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
x = np.arange(len(ORDER))
w = 0.38
print("\nedit_accuracy_clean_vs_erroneous: fix rate of the first resample (Thought-ICS, regraded)")
top = 0
for i, m in enumerate(ORDER):
    R = [v["thought"] for v in paired if v["thought"]["model"] == m and v["thought"]["cat"] != "failed"]
    fixed = lambda r: ok(r.get("answer"), r.get("ground_truth"), r["dataset"], r["fixed"])
    for off, sel, col in ((-w / 2, ("exact", "earlier"), EXACT), (w / 2, ("later",), ERRONEOUS)):
        S = [r for r in R if r["cat"] in sel]
        k = sum(fixed(r) for r in S)
        rate = 100 * k / max(1, len(S))
        lo, hi = wilson(k, len(S))
        top = max(top, hi)
        ax.bar(i + off, rate, w * 0.94, color=col, zorder=2)
        if k == 0:
            ax.text(i + off, 0.4, "0", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.errorbar(i + off, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        print(f"  {m:10s} {'clean' if len(sel) == 2 else 'erroneous':9s} {rate:5.1f}% ({k}/{len(S)}) [{lo:4.1f}, {hi:4.1f}]")
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=EXACT, label="Clean prefix (self ≤ oracle)"),
                   Patch(color=ERRONEOUS, label="Erroneous prefix (self > oracle)")], loc="upper left", fontsize=10)
ax.set_xticks(x, [SHORT[m] for m in ORDER], fontsize=11)
ax.set_xlabel("Model")
ax.set_ylabel("Correction accuracy (%)")
ax.set_ylim(0, math.ceil((top + 8) / 10) * 10)
save(fig, "edit_accuracy_clean_vs_erroneous")

# ---- Fig: localization_error_distribution (Thought-ICS) ----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
xs = np.linspace(-5, 10, 601)
bw = 0.45
print("\nlocalization_error_distribution: self - oracle (Thought-ICS, usable localizations)")
ymax = 0
curves = {}
for _, fam, ms in FAMILIES:
    for m in ms:
        dev = np.array([v["thought"]["step"] - v["thought"]["oracle"] for v in paired
                        if v["thought"]["model"] == m and v["thought"]["cat"] != "failed"], float)
        dens = np.exp(-0.5 * ((xs[:, None] - dev[None, :]) / bw) ** 2).sum(1) / (len(dev) * bw * math.sqrt(2 * math.pi))
        curves[m] = (dev, dens)
        ymax = max(ymax, dens.max())
for ax, (title, fam, ms) in zip(axes, FAMILIES):
    ax.axvline(0, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
    for m, col in zip(ms, FAMILY[fam]):
        dev, dens = curves[m]
        ax.fill_between(xs, dens, color=col, alpha=0.08, lw=0, zorder=1)
        ax.plot(xs, dens, color=col, lw=2, zorder=2,
                label=f"{LONG[m].split('-')[-1]}  (exact {100 * np.mean(dev == 0):.0f}%, mean {dev.mean():+.1f})")
        print(f"  {m:10s} n={len(dev):3d} exact {100*np.mean(dev==0):4.0f}%  mean {dev.mean():+.2f}  median {np.median(dev):+.0f}")
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=6)
    ax.set_xlim(-5, 10)
    ax.set_xticks([-4, 0, 4, 8])
    ax.legend(loc="upper right", fontsize=9, handlelength=1.6)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Density")
axes[1].set_xlabel("Localization deviation, self − oracle (thought steps)")
axes[0].set_ylim(0, ymax * 1.28)
axes[2].text(-0.7, ymax * 0.62, "← earlier\n(clean)", ha="right", va="center", fontsize=10, color=EXACT)
axes[2].text(0.7, ymax * 0.62, "later →\n(erroneous)", ha="left", va="center", fontsize=10, color=ERRONEOUS)
fig.tight_layout(w_pad=1.2)
save(fig, "localization_error_distribution")


# ---- Fig: self-verification by correction iteration (Thought-ICS, self-verified) ---------
# (file name kept from the paper; grouping is by verification call n, not by terminal iteration:
#  every response still in the correction loop is verified at iteration n, and recall /
#  specificity / accuracy are computed over those verifications)
SPEC, RECALL, ACC = "#6250d6", "#bf7f00", INK2
calls = defaultdict(list)  # n -> [(says_correct, actually_correct)]
for d in sorted(glob.glob(str(EXP / "full" / "*_selfverify_shared_fbattempt_*/"))):
    d = Path(d)
    c = json.load(open(d / "config.json"))
    for line in open(d / "thought_results.jsonl"):
        r = json.loads(line)
        its = r["iterations"]
        cur = ok(its[0]["answer"], r["ground_truth"], c["dataset"], its[0]["correct"])
        n = 0
        for it in its[1:]:
            v = it.get("verification") or it.get("localization", {}).get("verification")
            if v is not None:
                n += 1
                calls[min(n, 10)].append((v["verdict"] is True, cur))
            if "gen_status" in it:
                cur = ok(it["answer"], r["ground_truth"], c["dataset"], it["correct"])
ns = sorted(calls)
rec = [100 * sum(not s for s, a in calls[n] if not a) / max(1, sum(not a for _, a in calls[n])) for n in ns]
spe = [100 * sum(s for s, a in calls[n] if a) / max(1, sum(a for _, a in calls[n])) for n in ns]
acc = [100 * sum(s == a for s, a in calls[n]) / len(calls[n]) for n in ns]
print("\nself-verification by correction iteration (Thought-ICS, all models)")
for n, a, b, c_, k in zip(ns, spe, rec, acc, [len(calls[n]) for n in ns]):
    print(f"  n={n:2d} calls {k:5d}  specificity {a:5.1f}  recall {b:5.1f}  accuracy {c_:5.1f}")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ns, spe, color=SPEC, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.2, label="Specificity", zorder=3)
ax.plot(ns, rec, color=RECALL, lw=2, marker="^", ms=7.5, mec=SURFACE, mew=1.2, label="Recall", zorder=3)
ax.plot(ns, acc, color=ACC, lw=1.6, ls=(0, (4, 2.5)), marker="s", ms=6, mec=SURFACE, mew=1.2, label="Accuracy", zorder=2)
ax.set_xlabel("Correction iteration")
ax.set_ylabel("Percentage (%)")
ax.set_xticks(ns)
ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False)
ax.legend(loc="center right", fontsize=11)
save(fig, "self_verification_by_terminal_iteration_part1a")
