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


# ---- Fig: self-verification metrics by model (appendix; Thought-ICS, all verification calls) ----
bym = defaultdict(list)
for d in sorted(glob.glob(str(EXP / "full" / "*_selfverify_shared_fbattempt_*/"))):
    d = Path(d)
    c = json.load(open(d / "config.json"))
    for line in open(d / "thought_results.jsonl"):
        r = json.loads(line)
        its = r["iterations"]
        cur = ok(its[0]["answer"], r["ground_truth"], c["dataset"], its[0]["correct"])
        for it in its[1:]:
            v = it.get("verification") or it.get("localization", {}).get("verification")
            if v is not None:
                bym[c["model"]].append((v["verdict"] is True, cur))
            if "gen_status" in it:
                cur = ok(it["answer"], r["ground_truth"], c["dataset"], it["correct"])
FAM_ORDER = [m for _, _, ms in FAMILIES for m in ms]
fig, ax = plt.subplots(figsize=(12, 6))
w = 0.26
print("\nself-verification by model (Thought-ICS, all calls)")
for i, m in enumerate(FAM_ORDER):
    R = bym[m]
    vals = (100 * sum(not s for s, a in R if not a) / max(1, sum(not a for _, a in R)),
            100 * sum(s for s, a in R if a) / max(1, sum(a for _, a in R)),
            100 * sum(s == a for s, a in R) / len(R))
    print(f"  {m:11s} calls {len(R):5d}  recall {vals[0]:5.1f}  specificity {vals[1]:5.1f}  accuracy {vals[2]:5.1f}")
    for j, (val, col) in enumerate(zip(vals, (RECALL, SPEC, ACC))):
        x = i + (j - 1) * w
        ax.bar(x, val, w * 0.92, color=col, zorder=2)
        ax.text(x, val + 1.2, f"{val:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
for b in (2.5, 5.5):
    ax.axvline(b, color=AXIS, lw=1, zorder=1)
for xc, name in ((1, "Llama"), (4, "Qwen"), (6.5, "gpt-oss")):
    ax.text(xc, 104, name, ha="center", va="bottom", fontsize=12, color=INK2)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=RECALL, label="Recall"), Patch(color=SPEC, label="Specificity"),
                   Patch(color=ACC, label="Accuracy")], loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12)
ax.set_xticks(range(len(FAM_ORDER)), [LONG[m] for m in FAM_ORDER], fontsize=11)
ax.set_ylabel("Self-verification metric (%)")
ax.set_ylim(0, 110)
ax.set_yticks(range(0, 101, 20))
ax.grid(axis="x", visible=False)
save(fig, "self_verification_by_model")


# ---- Fig: accuracy over correction iterations, with vs without carried feedback (Thought-ICS) ----
# (file name kept from the paper's confidence-safeguard figure). Accuracy over ALL problems after n
# correction iterations: each problem counts with its current answer (a stopped problem keeps its
# final answer). Only (model, dataset) pairs where every plotted run exists are used.
FEEDBACK_VARIANT = "all"          # which carried-feedback variant the paper presents
NOFB, WITHFB = "#86b6ef", "#256abf"


def trajectories(pattern):
    """{(model, dataset_tag): {pid: [correct after n=0..10]}} for the Thought-ICS runs matching pattern."""
    out = {}
    for d in sorted(glob.glob(str(EXP / "full" / pattern))):
        d = Path(d)
        if not (d / "thought_results.jsonl").exists():
            continue
        c = json.load(open(d / "config.json"))
        key = (c["model"], d.name.split("_selfverify")[0])
        T = {}
        for line in open(d / "thought_results.jsonl"):
            r = json.loads(line)
            gens = {it["iteration"]: ok(it["answer"], r["ground_truth"], c["dataset"], it["correct"])
                    for it in r["iterations"] if "gen_status" in it}
            cur, traj = gens[0], []
            for n in range(11):
                cur = gens.get(n, cur)
                traj.append(cur)
            T[r["problem_id"]] = traj
        out[key] = T  # latest run per (model, dataset) wins (sorted by timestamp)
    return out


S = trajectories("*_selfverify_shared_fbattempt_*/")
FB = trajectories(f"*_selfverifyfb{FEEDBACK_VARIANT}_shared_fbattempt_*/")
keys = sorted(set(S) & set(FB)) if FB else sorted(S)
print(f"\naccuracy over correction iterations: {len(keys)} (model, dataset) pairs"
      f"{'' if FB else ' -- no ' + FEEDBACK_VARIANT + ' runs yet, plotting without-feedback only'}")
acc = lambda T: [100 * np.mean([T[k][p][n] for k in keys for p in T[k]]) for n in range(11)]
s_curve = acc(S)
fig, ax = plt.subplots(figsize=(6, 4))
xs = list(range(11))
ax.plot(xs, [s_curve[0]] * 11, color=MUTED, lw=1.4, ls=(0, (4, 2.5)), label="Initial", zorder=1)
ax.plot(xs, s_curve, color=NOFB, lw=2, marker="o", ms=6.5, mec=SURFACE, mew=1.2, label="Without feedback", zorder=2)
if FB:
    f_curve = acc(FB)
    ax.plot(xs, f_curve, color=WITHFB, lw=2, marker="^", ms=7, mec=SURFACE, mew=1.2, label="With feedback", zorder=3)
    print("  with feedback   :", " ".join(f"{v:5.1f}" for v in f_curve))
print("  without feedback:", " ".join(f"{v:5.1f}" for v in s_curve))
ax.set_xlabel("Correction iteration")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(xs)
ax.grid(axis="x", visible=False)
ax.legend(loc="lower right", fontsize=11)
save(fig, "self_verification_by_terminal_iteration_part1b")


# ---- Fig (appendix): oracle localization agreement across the three judges ----------------
JUDGES = [("opus", "Claude Opus"), ("sonnet", "Claude Sonnet"), ("haiku", "Claude Haiku")]
AGREE = [("Unanimous", "#184f95"), ("Within ±1", "#3987e5"), ("Within ±2", "#86b6ef"), ("Further apart", FAILED)]
items, labels = {}, defaultdict(dict)
for o in ("oracle_full", "oracle_slm"):
    for f in glob.glob(str(EXP / o / "batch_*.json")):
        for x in json.load(open(f)):
            items[x["id"]] = len(x["steps"])
    for f in glob.glob(str(EXP / o / "labels_*_*.json")):
        j = Path(f).name.split("_")[1]
        for x in json.load(open(f)):
            labels[x["id"]][j] = int(x["first_error_step"])
cat = {}
for i, v in labels.items():
    s = list(v.values())
    spread = max(s) - min(s)
    cat[i] = 0 if spread == 0 else 1 if spread <= 1 else 2 if spread <= 2 else 3
n_items = len(cat)
shares = [100 * sum(c <= k for c in cat.values()) / n_items for k in range(3)]
print(f"\noracle agreement: {n_items} items; unanimous {shares[0]:.0f}%  within±1 {shares[1]:.0f}%  within±2 {shares[2]:.0f}%")
fig, ax = plt.subplots(figsize=(8, 5))
rnd = np.random.default_rng(0)
for k, (j, name) in enumerate(JUDGES):
    pos = [((labels[i][j] - 1) / max(1, items[i] - 1), cat[i]) for i in labels if labels[i].get(j, 0) > 0 and i in items]
    ys = np.array([p for p, _ in pos])
    vp = ax.violinplot([ys], positions=[k], widths=0.8, showextrema=False, showmedians=False)
    for b in vp["bodies"]:
        b.set_facecolor("#e1e0d9")
        b.set_edgecolor(AXIS)
        b.set_alpha(1)
    for ci in (3, 2, 1, 0):  # draw the most-agreed points last (on top)
        sel = np.array([c == ci for _, c in pos])
        if sel.any():
            ax.scatter(k + rnd.uniform(-0.18, 0.18, sel.sum()), ys[sel], s=7, color=AGREE[ci][1], alpha=0.55,
                       lw=0, zorder=3)
    ax.hlines(np.median(ys), k - 0.3, k + 0.3, color=INK, lw=1.4, zorder=4)
from matplotlib.lines import Line2D  # noqa: E402
ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=6, color=c, label=f"{n} ({sh:.0f}%)" if idx < 3 else n)
                   for idx, ((n, c), sh) in enumerate(zip(AGREE, shares + [0]))] +
          [Line2D([], [], color=INK, lw=1.4, label="Median")], loc="upper right", fontsize=10)
ax.set_xticks(range(len(JUDGES)), [n for _, n in JUDGES])
ax.set_xlabel("Oracle model")
ax.set_ylabel("Normalized error position")
ax.set_ylim(-0.04, 1.04)
ax.set_xlim(-0.6, len(JUDGES) - 0.4 + 0.9)
ax.grid(axis="x", visible=False)
save(fig, "oracle_localization_agreement")

# ---- Fig (appendix): Token-ICS fix rate after clean vs erroneous prefix ---------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
x = np.arange(len(ORDER))
w = 0.38
top = 0
print("\nedit_accuracy_clean_vs_erroneous_token: fix rate of the first resample (Token-ICS, regraded)")
pool = {"clean": [0, 0], "erroneous": [0, 0]}
for i, m in enumerate(ORDER):
    R = [v["token"] for v in paired if v["token"]["model"] == m and v["token"]["cat"] != "failed"]
    fixed = lambda r: ok(r.get("answer"), r.get("ground_truth"), r["dataset"], r["fixed"])
    for off, sel, col, nm in ((-w / 2, ("exact", "earlier"), EXACT, "clean"), (w / 2, ("later",), ERRONEOUS, "erroneous")):
        S = [r for r in R if r["cat"] in sel]
        k = sum(fixed(r) for r in S)
        pool[nm][0] += k
        pool[nm][1] += len(S)
        rate = 100 * k / max(1, len(S))
        lo, hi = wilson(k, len(S))
        top = max(top, hi)
        ax.bar(i + off, rate, w * 0.94, color=col, zorder=2)
        ax.errorbar(i + off, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        if k == 0:
            ax.text(i + off, 0.4, "0", ha="center", va="bottom", fontsize=8, color=INK2)
print("  pooled:", {k: f"{100*a/max(1,b):.1f}% ({a}/{b})" for k, (a, b) in pool.items()})
ax.legend(handles=[Patch(color=EXACT, label="Clean prefix (self ≤ oracle)"),
                   Patch(color=ERRONEOUS, label="Erroneous prefix (self > oracle)")], loc="upper left", fontsize=10)
ax.set_xticks(x, [SHORT[m] for m in ORDER], fontsize=11)
ax.set_xlabel("Model")
ax.set_ylabel("Correction accuracy (%)")
ax.set_ylim(0, math.ceil((top + 8) / 10) * 10)
save(fig, "edit_accuracy_clean_vs_erroneous_token")


# ---- Fig (appendix): confusion-matrix components by correction iteration (Thought-ICS) -------
# share of verifications at iteration n that are TP / FP / TN / FN (positive = flagged incorrect);
# amber = incorrect responses (TP, FN; recall), violet = correct responses (TN, FP; specificity);
# solid = right verdict, dashed = wrong verdict
CM = [("TP (caught error)", lambda s, a: not s and not a, "#a86f00", "-", "o"),
      ("FP (false alarm)", lambda s, a: not s and a, "#8f83ea", (0, (4, 2.5)), "s"),
      ("TN (correctly left alone)", lambda s, a: s and a, "#4a3aa7", "-", "^"),
      ("FN (missed error)", lambda s, a: s and not a, "#d9a031", (0, (4, 2.5)), "D")]
fig, ax = plt.subplots(figsize=(8, 5))
print("\nconfusion matrix by correction iteration (% of verifications at n)")
for name, f, col, ls, mk in CM:
    ys = [100 * sum(f(s, a) for s, a in calls[n]) / len(calls[n]) for n in ns]
    ax.plot(ns, ys, color=col, lw=2, ls=ls, marker=mk, ms=6.5, mec=SURFACE, mew=1.2, label=name, zorder=3)
    print(f"  {name:28s}", " ".join(f"{y:5.1f}" for y in ys))
ax.set_xlabel("Correction iteration")
ax.set_ylabel("Rate (%)")
ax.set_xticks(ns)
ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=11, ncol=2)
save(fig, "confusion_matrix_by_iteration")


# ---- Fig (appendix): chain and thought length distributions (Thought-ICS first attempts) ------
import os  # noqa: E402
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402
sys.path.insert(0, str(REPO))
from thought_ics.budget_study.run_study import MODELS as _M  # noqa: E402

chain_len, thought_tok, by_correct = defaultdict(list), defaultdict(list), {True: [], False: []}
for m in FAM_ORDER:
    tok = AutoTokenizer.from_pretrained(_M[m]["hf"])
    for d in sorted(glob.glob(str(EXP / "full" / f"{m}_*_L2_shared_fbattempt_*/"))):
        d = Path(d)
        c = json.load(open(d / "config.json"))
        for line in open(d / "thought_results.jsonl"):
            r = json.loads(line)
            it0 = r["iterations"][0]
            ch = it0["chain"]
            if not ch:
                continue
            chain_len[m].append(len(ch))
            thought_tok[m] += [len(t) for t in tok(ch, add_special_tokens=False)["input_ids"]]
            by_correct[ok(it0["answer"], r["ground_truth"], c["dataset"], it0["correct"])].append(len(ch))


def kde(vals, xs, bw):
    v = np.asarray(vals, float)
    return np.exp(-0.5 * ((xs[:, None] - v[None, :]) / bw) ** 2).sum(1) / (len(v) * bw * math.sqrt(2 * math.pi))


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
xc, xt = np.linspace(0, 60, 601), np.linspace(0, 250, 626)
print("\nchain / thought lengths (Thought-ICS first attempts)")
for _, fam, ms in FAMILIES:
    for m, col in zip(ms, FAMILY[fam]):
        cl, tl = chain_len[m], thought_tok[m]
        axes[0].plot(xc, kde(cl, xc, 2.0), color=col, lw=2, label=f"{LONG[m]} (median {np.median(cl):.0f})")
        axes[1].plot(xt, kde(tl, xt, 8.0), color=col, lw=2, label=f"{LONG[m]} (median {np.median(tl):.0f})")
        print(f"  {m:11s} chains {len(cl):4d}  median steps {np.median(cl):5.0f} (mean {np.mean(cl):5.1f})  median tokens/thought {np.median(tl):4.0f} (mean {np.mean(tl):4.0f})")
for ax, xl, title in ((axes[0], "Number of thought steps", "Chain length by model"),
                      (axes[1], "Tokens per thought", "Thought length by model")):
    ax.set_xlabel(xl)
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=13, loc="left", color=INK)
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=9.5)
axes[0].set_xlim(0, 60)
axes[1].set_xlim(0, 250)
fig.tight_layout(w_pad=2)
save(fig, "chain_thought_length_by_model")

fig, ax = plt.subplots(figsize=(8, 5))
for k, col, nm in ((True, EXACT, "Correct"), (False, ERRONEOUS, "Incorrect")):
    y = kde(by_correct[k], xc, 2.0)
    ax.fill_between(xc, y, color=col, alpha=0.15, lw=0)
    ax.plot(xc, y, color=col, lw=2.2, label=f"{nm} (median {np.median(by_correct[k]):.0f})")
    print(f"  {nm:9s} chains {len(by_correct[k]):5d}  median steps {np.median(by_correct[k]):.0f} (mean {np.mean(by_correct[k]):.1f})")
ax.set_xlabel("Number of thought steps")
ax.set_ylabel("Density")
ax.set_xlim(0, 60)
ax.set_ylim(0, None)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=12)
save(fig, "chain_length_distribution")


# ---- Fig §5.3: self-correction lift in the fully autonomous setting -------------------------
# lift over the shared first attempt (final - first), paired 95% bootstrap CI over problems; a model
# is plotted only when Self-Refine, CoVe and Thought-ICS-F have all finished all six datasets.
SR_C, COVE_C, F_C = "#bf7f00", "#6250d6", "#c2457e"


def lift_rows():
    out = defaultdict(list)
    for sv in sorted({p.rsplit("_selfverify", 1)[0] for p in glob.glob(str(EXP / "full" / "*_selfverify_shared_fbattempt_*"))}):
        lat = lambda pat, f: ([d for d in sorted(glob.glob(pat)) if Path(d, f).exists()] or [None])[-1]
        b = lat(f"{sv}_baselines_*", "cove_results.jsonl")
        fa = lat(f"{sv}_selfverifyfb{FEEDBACK_VARIANT}_shared_fbattempt_*", "thought_results.jsonl")
        s = lat(f"{sv}_selfverify_shared_fbattempt_*", "thought_results.jsonl")
        if not (b and fa and s):
            continue
        c = json.load(open(Path(s) / "config.json"))
        L = lambda p: {json.loads(l)["problem_id"]: json.loads(l) for l in open(p)}
        SRr, CVr, Fr = L(Path(b) / "self_refine_results.jsonl"), L(Path(b) / "cove_results.jsonl"), L(Path(fa) / "thought_results.jsonl")
        for pid, r in L(Path(s) / "thought_results.jsonl").items():
            g = lambda it: ok(it["answer"], r["ground_truth"], c["dataset"], it["correct"])
            last = lambda rr: g([it for it in rr["iterations"] if "gen_status" in it][-1])
            out[c["model"]].append({"ds": c["dataset"], "init": g(r["iterations"][0]), "sr": g(SRr[pid]["iterations"][-1]),
                                    "cove": g(CVr[pid]["iterations"][-1]), "f": last(Fr[pid])})
    return out


LR = lift_rows()
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(ORDER))
w = 0.26
print("\nself_correction_lift_errorbar: lift over the first attempt [95% bootstrap CI]")
for j, (key, name, col) in enumerate((("sr", "Self-Refine", SR_C), ("cove", "CoVe", COVE_C), ("f", "Thought-ICS-F", F_C))):
    for i, m in enumerate(ORDER):
        R = LR.get(m, [])
        if len({r["ds"] for r in R}) < 6:
            continue
        rnd = random.Random(0)
        f = lambda S: 100 * sum(r[key] - r["init"] for r in S) / len(S)
        v = sorted(f([R[rnd.randrange(len(R))] for _ in R]) for _ in range(5000))
        est, lo, hi = f(R), v[125], v[4874]
        ax.bar(i + (j - 1) * w, est, w * 0.92, color=col, label=name if i == min(k for k, mm in enumerate(ORDER)
                                                                             if len({r['ds'] for r in LR.get(mm, [])}) == 6) else None, zorder=2)
        ax.errorbar(i + (j - 1) * w, est, yerr=[[est - lo], [hi - est]], fmt="none", ecolor=INK2, elinewidth=1.1, capsize=2.5, zorder=3)
        print(f"  {name:14s} {m:11s} {est:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]")
missing = [m for m in ORDER if len({r["ds"] for r in LR.get(m, [])}) < 6]
if missing:
    print("  not yet complete (omitted):", ", ".join(missing))
ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
ax.set_xticks(x, [SHORT[m] for m in ORDER])
ax.set_xlabel("Model")
ax.set_ylabel("Self-correction lift (%)")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=12)
save(fig, "self_correction_lift_errorbar")
