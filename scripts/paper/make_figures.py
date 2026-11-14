"""All of the paper's figures, from the runs in results/, drawn as in the original paper.

    python scripts/paper/make_figures.py [out_dir]          # default: paper_assets/figures

Main paper: l2_errorbar, self_localization_vs_oracle(_token), edit_accuracy_clean_vs_erroneous,
localization_error_distribution, self_verification_by_terminal_iteration_part1a (self-verification by
correction iteration), self_verification_by_terminal_iteration_part1b (with vs without feedback),
self_correction_lift_errorbar. Appendix: self_verification_by_model, confusion_matrix_by_iteration,
oracle_localization_agreement, edit_accuracy_clean_vs_erroneous_token, chain_thought_length_by_model,
chain_length_distribution. File names follow the paper's \\includegraphics paths.

Math benchmarks are regraded with Math-Verify; oracle-verified runs are truncated at the first correct
attempt (as in scripts/report_oracle_verified.py). Oracle localization labels: results/oracle/part{1,2}.
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

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper_assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- the original paper's colors, names and model orders ------------------------------------
# (from the scripts that drew the original figures; matplotlib defaults otherwise)
GREEN, BLUE, RED, GRAY = "#2ecc71", "#3498db", "#e74c3c", "#95a5a6"
GREEN_DK, RED_DK = "#27ae60", "#c0392b"
MODEL_COLORS = {"llama3b": "#e74c3c", "llama8b": "#c0392b", "llama70b": "#922b21", "qwen7b": "#3498db",
                "qwen14b": "#2980b9", "qwen32b": "#1a5276", "gptoss20b": "#27ae60", "gptoss120b": "#1e8449"}
ORDER = ["llama3b", "qwen7b", "llama8b", "qwen14b", "gptoss20b", "qwen32b", "llama70b", "gptoss120b"]   # by size
FAM_ORDER = ["llama3b", "llama8b", "llama70b", "qwen7b", "qwen14b", "qwen32b", "gptoss20b", "gptoss120b"]
SHORT = {"llama3b": "L3B", "qwen7b": "Q7B", "llama8b": "L8B", "qwen14b": "Q14B", "gptoss20b": "G20B",
         "qwen32b": "Q32B", "llama70b": "L70B", "gptoss120b": "G120B"}
LONG = {"llama3b": "LLaMA-3B", "llama8b": "LLaMA-8B", "llama70b": "LLaMA-70B", "qwen7b": "Qwen-7B",
        "qwen14b": "Qwen-14B", "qwen32b": "Qwen-32B", "gptoss20b": "OSS-20B", "gptoss120b": "OSS-120B"}
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
    fig.savefig(OUT / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


from scipy.interpolate import make_interp_spline  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402


# ---- Fig: l2_errorbar --------------------------------------------------------------------
def first_correct(rec, ds):
    """(init_ok, final_ok) under math_verify, run truncated at the first correct attempt."""
    for k, it in enumerate(rec["iterations"]):
        if "gen_status" in it and ok(it["answer"], rec["ground_truth"], ds, it["correct"]):
            return k == 0, True
    return False, False


rows_l2 = defaultdict(list)
for d in sorted(glob.glob(str(EXP / "*_oracle_2*/"))):
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


def lift_bars(ax, series, width):
    """Grouped bars in the original paper's style: translucent fill, same-color edge and error bars."""
    n = len(series)
    for k, (name, col, alpha, est) in enumerate(series):
        pos = np.arange(len(ORDER)) + (k - (n - 1) / 2) * width
        mid = np.array([e[0] for e in est])
        ax.bar(pos, mid, width, facecolor=col, edgecolor=col, alpha=alpha, label=name)
        ax.errorbar(pos, mid, yerr=[mid - [e[1] for e in est], [e[2] for e in est] - mid], fmt="none",
                    ecolor=col, capsize=3, capthick=1.5, elinewidth=1.5, alpha=0.8)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xticks(np.arange(len(ORDER)))
    ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=16)
    ax.set_xlabel("Model", fontsize=16)
    ax.set_ylabel("Self-Correction Lift (%)", fontsize=16)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=16)


print("\nl2_errorbar: lift over the shared first attempt [95% bootstrap CI]")
est = {j: [boot_lift(rows_l2[m], j) for m in ORDER] for j in (1, 2)}
for j, name in ((2, "Token-ICS"), (1, "Thought-ICS")):
    for m, e in zip(ORDER, est[j]):
        print(f"  {name:11s} {m:10s} {e[0]:5.1f} [{e[1]:5.1f}, {e[2]:5.1f}]  (n={len(rows_l2[m])})")
fig, ax = plt.subplots(figsize=(9, 6))
lift_bars(ax, [("Token-ICS", "#2ca02c", 0.7, est[2]), ("Thought-ICS", "#d62728", 0.9, est[1])], 0.28)
ax.legend(loc="upper left", fontsize=14, title="Under Oracle Verification:", title_fontsize=14,
          alignment="left")
ax.set_ylim(0, max(e[2] for j in est for e in est[j]) * 1.3)
plt.tight_layout()
save(fig, "l2_errorbar")

# ---- oracle rows ------------------------------------------------------------------------
orows = [r for o in ("part1", "part2") for r in json.load(open(EXP / "oracle" / o / "rows.json"))]
pair = defaultdict(dict)
for r in orows:
    pair[r["id"]][r["method"]] = r
paired = [v for v in pair.values() if "thought" in v and "token" in v]


# ---- Figs: self_localization_vs_oracle (Thought-ICS) and _token ---------------------------
def loc_fig(method, name):
    """Original construction (valid_localization_vs_oracle.py); failed localizations, which the
    original runs did not have, are stacked in gray on the erroneous bar."""
    models = FAM_ORDER
    S = {}
    print(f"\n{name}: first localization vs oracle ({method})")
    for m in models:
        R = [v[method] for v in paired if v[method]["model"] == m]
        n = len(R)
        cnt = {k: sum(r["cat"] == k for r in R) for k in ("exact", "earlier", "later", "failed")}
        p = {k: 100 * v / n for k, v in cnt.items()}
        clo, chi = wilson(cnt["exact"] + cnt["earlier"], n)
        llo, lhi = wilson(cnt["later"], n)
        S[m] = (p, p["exact"] + p["earlier"], clo, chi, llo, lhi)
        print(f"  {m:10s} n={n:3d} exact {p['exact']:4.0f} earlier {p['earlier']:4.0f} | later {p['later']:4.0f} "
              f"failed {p['failed']:4.0f} | clean {p['exact'] + p['earlier']:4.0f}")
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6, 4.5))
    exact = [S[m][0]["exact"] for m in models]
    earlier = [S[m][0]["earlier"] for m in models]
    later = [S[m][0]["later"] for m in models]
    failed = [S[m][0]["failed"] for m in models]
    clean = [S[m][1] for m in models]
    ax.bar(x - width / 2, exact, width, color=GREEN, alpha=0.85, label="Clean Prefix (self ≤ oracle)")
    ax.bar(x - width / 2, earlier, width, bottom=exact, color=BLUE, alpha=0.85)
    ax.bar(x + width / 2, later, width, color=RED, alpha=0.85, label="Erroneous Prefix (self > oracle)")
    if any(failed):
        ax.bar(x + width / 2, failed, width, bottom=later, color=GRAY, alpha=0.85, label="Failed Localization")
    ax.errorbar(x - width / 2, clean, yerr=[[c - S[m][2] for c, m in zip(clean, models)],
                                            [S[m][3] - c for c, m in zip(clean, models)]],
                fmt="none", ecolor=GREEN_DK, capsize=3, capthick=1.5, elinewidth=1.5)
    ax.errorbar(x + width / 2, later, yerr=[[l - S[m][4] for l, m in zip(later, models)],
                                            [S[m][5] - l for l, m in zip(later, models)]],
                fmt="none", ecolor=RED_DK, capsize=3, capthick=1.5, elinewidth=1.5)
    for i, m in enumerate(models):
        ax.annotate(f"{clean[i]:.0f}%", xy=(x[i] - width / 2, S[m][3] + 1.5), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", rotation=90, annotation_clip=False)
    ax.set_ylabel("Percentage (%)", fontsize=14)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in models], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=11, ncol=1, framealpha=0.9)
    ax.set_ylim(0, 120)
    ax.set_yticks(range(0, 101, 20))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save(fig, name)


loc_fig("thought", "self_localization_vs_oracle")
loc_fig("token", "self_localization_vs_oracle_token")


# ---- Figs: edit_accuracy_clean_vs_erroneous (Thought-ICS) and _token ----------------------
def edit_fig(method, name):
    """Fix rate of the first resample after a clean vs erroneous prefix (original construction)."""
    models = FAM_ORDER
    fixed = lambda r: ok(r.get("answer"), r.get("ground_truth"), r["dataset"], r["fixed"])
    acc, lo_, hi_ = {"c": [], "e": []}, {"c": [], "e": []}, {"c": [], "e": []}
    pool = {"c": [0, 0], "e": [0, 0]}
    print(f"\n{name}: fix rate of the first resample ({method}, regraded)")
    for m in models:
        R = [v[method] for v in paired if v[method]["model"] == m and v[method]["cat"] != "failed"]
        for key, sel in (("c", ("exact", "earlier")), ("e", ("later",))):
            Sx = [r for r in R if r["cat"] in sel]
            k = sum(fixed(r) for r in Sx)
            a = 100 * k / len(Sx) if Sx else 0
            lo, hi = wilson(k, len(Sx))
            acc[key].append(a)
            lo_[key].append(a - lo)
            hi_[key].append(hi - a)
            pool[key][0] += k
            pool[key][1] += len(Sx)
            print(f"  {m:10s} {'clean' if key == 'c' else 'erroneous':9s} {a:5.1f}% ({k}/{len(Sx)})")
    print("  pooled:", {("clean" if k == "c" else "erroneous"): f"{100 * a / max(1, b):.1f}% ({a}/{b})"
                        for k, (a, b) in pool.items()})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bar_width = 0.38
    x = np.arange(len(models))
    ax.bar(x - bar_width / 2, acc["c"], bar_width, yerr=[lo_["c"], hi_["c"]], label="Clean Prefix (self ≤ oracle)",
           color=GREEN, alpha=0.85, capsize=3, error_kw={"linewidth": 1.5, "ecolor": GREEN_DK})
    ax.bar(x + bar_width / 2, acc["e"], bar_width, yerr=[lo_["e"], hi_["e"]], label="Erroneous Prefix (self > oracle)",
           color=RED, alpha=0.85, capsize=3, error_kw={"linewidth": 1.5, "ecolor": RED_DK})
    top = max(a + h for key in ("c", "e") for a, h in zip(acc[key], hi_[key]))
    pad = top * 0.012
    for i in range(len(models)):
        for key, off in (("c", -bar_width / 2), ("e", bar_width / 2)):
            if acc[key][i] > 0:
                ax.annotate(f"{acc[key][i]:.1f}%", xy=(x[i] + off, acc[key][i] + hi_[key][i] + pad), ha="center",
                            va="bottom", fontsize=9, fontweight="bold", rotation=90)
    ax.set_ylabel("Edit Accuracy (%)", fontsize=14)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in models], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, math.ceil(top * 1.3 / 5) * 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=11, ncol=1, framealpha=0.9)
    plt.tight_layout()
    save(fig, name)


edit_fig("thought", "edit_accuracy_clean_vs_erroneous")
edit_fig("token", "edit_accuracy_clean_vs_erroneous_token")

# ---- Fig: localization_error_distribution (Thought-ICS; localization_error_distribution.py) -
# same family hues as MODEL_COLORS, lighter for smaller models
DEV_COLORS = {"llama3b": "#f1948a", "llama8b": "#e74c3c", "llama70b": "#922b21", "qwen7b": "#85c1e9",
              "qwen14b": "#3498db", "qwen32b": "#1a5276", "gptoss20b": "#58d68d", "gptoss120b": "#1e8449"}
print("\nlocalization_error_distribution: self - oracle (Thought-ICS, usable localizations)")
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.arange(-5, 12) - 0.5
bin_centers = np.arange(-5, 11)
x_smooth = np.linspace(bin_centers[0], bin_centers[-1], 300)
for m in FAM_ORDER:
    devs = np.array([v["thought"]["step"] - v["thought"]["oracle"] for v in paired
                     if v["thought"]["model"] == m and v["thought"]["cat"] != "failed"])
    counts, _ = np.histogram(devs, bins=bins, density=True)
    mean_dev = int(round(np.mean(devs)))
    counts_smooth = np.maximum(make_interp_spline(bin_centers, counts, k=3)(x_smooth), 0)
    ax.plot(x_smooth, counts_smooth, color=DEV_COLORS[m], linewidth=2.5, label=f"{LONG[m]} (μ={mean_dev:+d})")
    print(f"  {m:10s} n={len(devs):3d} exact {100 * np.mean(devs == 0):4.0f}%  mean {devs.mean():+.2f}  "
          f"median {np.median(devs):+.0f}")
ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
y_max = ax.get_ylim()[1]
ax.text(-2.5, y_max * 0.85, "← Earlier\n(clean prefix)", ha="center", va="top", fontsize=18, alpha=0.7, color="green")
ax.text(3, y_max * 0.85, "Later →\n(erroneous prefix)", ha="center", va="top", fontsize=18, alpha=0.7, color="red")
ax.set_xlabel("Localization Deviation (thought steps)", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.set_xlim(-5, 10)
ax.tick_params(axis="both", labelsize=12)
ax.legend(fontsize=14, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save(fig, "localization_error_distribution")


# ---- self-verification calls (Thought-ICS, self-verified runs) -----------------------------
# Every response still in the correction loop is verified at correction iteration n.
METRIC_COLORS = {"recall": "#2ecc71", "specificity": "#e74c3c", "accuracy": "#3498db"}
calls = defaultdict(list)  # n -> [(says_correct, actually_correct)]
bym = defaultdict(list)    # model -> [(says_correct, actually_correct)]
for d in sorted(glob.glob(str(EXP / "*_self_2*/"))):
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
                bym[c["model"]].append((v["verdict"] is True, cur))
            if "gen_status" in it:
                cur = ok(it["answer"], r["ground_truth"], c["dataset"], it["correct"])
recall = lambda C: 100 * sum(not s for s, a in C if not a) / max(1, sum(not a for _, a in C))
specificity = lambda C: 100 * sum(s for s, a in C if a) / max(1, sum(a for _, a in C))
accuracy = lambda C: 100 * sum(s == a for s, a in C) / max(1, len(C))

# ---- Fig: self_verification_by_terminal_iteration_part1a (self_verification_analysis.py) ---
ns = sorted(calls)
spe, rec, acc = ([f(calls[n]) for n in ns] for f in (specificity, recall, accuracy))
print("\nself-verification by correction iteration (Thought-ICS, all models)")
for n, a, b, c_ in zip(ns, spe, rec, acc):
    print(f"  n={n:2d} calls {len(calls[n]):5d}  specificity {a:5.1f}  recall {b:5.1f}  accuracy {c_:5.1f}")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(ns, spe, marker="o", linewidth=2, markersize=8, label="Specificity", color=METRIC_COLORS["specificity"])
ax.plot(ns, rec, marker="o", linewidth=2, markersize=8, label="Recall", color=METRIC_COLORS["recall"])
ax.plot(ns, acc, marker="s", linewidth=2, markersize=8, label="Accuracy", color=METRIC_COLORS["accuracy"])
ax.set_xlabel("Correction Iteration", fontsize=14)
ax.set_ylabel("Percentage (%)", fontsize=14)
ax.legend(loc="center right", fontsize=12)
ax.set_xlim(0.5, 10.5)
allm = spe + rec + acc
ax.set_ylim(max(0, min(allm) - 5), min(100, max(allm) + 5))
ax.set_xticks(range(1, 11))
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "self_verification_by_terminal_iteration_part1a")

# ---- Fig (appendix): self_verification_by_model (self_verification_analysis.py) ------------
print("\nself-verification by model (Thought-ICS, all calls)")
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.25
for i, metric in enumerate(("recall", "specificity", "accuracy")):
    f = {"recall": recall, "specificity": specificity, "accuracy": accuracy}[metric]
    values = [f(bym[m]) for m in FAM_ORDER]
    bars = ax.bar([xi + (i - 1) * width for xi in range(len(FAM_ORDER))], values, width,
                  label=metric.capitalize(), color=METRIC_COLORS[metric], alpha=0.8)
    for bar, val in zip(bars, values):
        ax.annotate(f"{val:.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
for m in FAM_ORDER:
    print(f"  {m:11s} calls {len(bym[m]):5d}  recall {recall(bym[m]):5.1f}  specificity {specificity(bym[m]):5.1f}"
          f"  accuracy {accuracy(bym[m]):5.1f}")
ax.set_xlabel("Model", fontsize=18)
ax.set_ylabel("Self-Verification Metric (%)", fontsize=18)
ax.set_xticks(range(len(FAM_ORDER)))
ax.set_xticklabels([LONG[m] for m in FAM_ORDER], fontsize=14)
ax.legend(loc="upper center", fontsize=12, ncol=3)
ax.set_ylim(0, 115)
ax.set_yticks(range(0, 101, 20))
ax.tick_params(axis="y", labelsize=14)
ax.grid(axis="y", alpha=0.3)
for b in (3, 6):
    ax.axvline(x=b - 0.5, color="black", linestyle="-", linewidth=1.5, alpha=0.7)
plt.tight_layout()
save(fig, "self_verification_by_model")

# ---- Fig (appendix): confusion_matrix_by_iteration (self_verification_analysis.py) ---------
cm_colors = {"TP": "#2ecc71", "FP": "#e74c3c", "TN": "#3498db", "FN": "#f39c12"}
CM = [("TP", "TP (caught error)", lambda s, a: not s and not a, "o"),
      ("FP", "FP (false alarm)", lambda s, a: not s and a, "s"),
      ("TN", "TN (correctly left alone)", lambda s, a: s and a, "^"),
      ("FN", "FN (missed error)", lambda s, a: s and not a, "d")]
print("\nconfusion matrix by correction iteration (% of verifications at n)")
fig, ax = plt.subplots(figsize=(8, 5))
for key, label, f, mk in CM:
    ys = [100 * sum(f(s, a) for s, a in calls[n]) / len(calls[n]) for n in ns]
    ax.plot(ns, ys, marker=mk, linewidth=2, markersize=8, label=label, color=cm_colors[key])
    print(f"  {label:28s}", " ".join(f"{y:5.1f}" for y in ys))
ax.set_xlabel("Iteration", fontsize=16)
ax.set_ylabel("Rate (%)", fontsize=16)
ax.legend(loc="upper right", fontsize=11)
ax.set_xlim(0.5, 10.5)
ax.set_ylim(0, 100)
ax.set_xticks(range(1, 11))
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
save(fig, "confusion_matrix_by_iteration")


# ---- Fig: self_verification_by_terminal_iteration_part1b -----------------------------------
# Accuracy over all problems after n correction iterations, Thought-ICS without vs with feedback carried
# through correction (each problem counts with its current answer; a stopped problem keeps its final
# answer). Only (model, dataset) pairs where both runs exist are used.
FEEDBACK_VARIANT = "all"          # Thought-ICS-F


def trajectories(pattern):
    """{(model, dataset_tag): {pid: [correct after n=0..10]}} for the Thought-ICS runs matching pattern."""
    out = {}
    for d in sorted(glob.glob(str(EXP / pattern))):
        d = Path(d)
        if not (d / "thought_results.jsonl").exists():
            continue
        c = json.load(open(d / "config.json"))
        key = (c["model"], d.name.split("_self")[0])
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


S = trajectories("*_self_2*/")
FB = trajectories(f"*_self-fb-{FEEDBACK_VARIANT}_2*/")
keys = sorted(set(S) & set(FB)) if FB else sorted(S)
print(f"\naccuracy over correction iterations: {len(keys)} (model, dataset) pairs"
      f"{'' if FB else ' -- no ' + FEEDBACK_VARIANT + ' runs yet, plotting without-feedback only'}")
curve = lambda T: [100 * np.mean([T[k][p][n] for k in keys for p in T[k]]) for n in range(11)]
s_curve = curve(S)
its = list(range(1, 11))
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(its, [s_curve[0]] * 10, marker="o", linewidth=2, markersize=8, label="Initial", color="#1f77b4")
ax.plot(its, s_curve[1:], marker="s", linewidth=2, markersize=8, label="Final (without feedback)", color="#ff7f0e")
allv = [s_curve[0]] + s_curve[1:]
if FB:
    f_curve = curve(FB)
    ax.plot(its, f_curve[1:], marker="^", linewidth=2, markersize=8, label="Final (with feedback)", color="#d62728")
    allv += f_curve[1:]
    print("  with feedback   :", " ".join(f"{v:5.1f}" for v in f_curve))
print("  without feedback:", " ".join(f"{v:5.1f}" for v in s_curve))
ax.set_xlabel("Correction Iteration", fontsize=14)
ax.set_ylabel("Accuracy (%)", fontsize=14)
ax.legend(loc="upper right", fontsize=11)
ax.set_xlim(0.5, 10.5)
ax.set_ylim(max(0, min(allv) - 5), min(100, max(allv) + 5))
ax.set_xticks(its)
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "self_verification_by_terminal_iteration_part1b")


# ---- Fig (appendix): oracle_localization_agreement ----------------------------------------
JUDGES = [("opus", "Claude Opus"), ("sonnet", "Claude Sonnet"), ("haiku", "Claude Haiku")]
VIOLIN = ["#8cb0fa", "#8be3c0", "#f8d05b"]       # the original figure's violin fills
items, labels = {}, defaultdict(dict)
for o in ("part1", "part2"):
    for f in glob.glob(str(EXP / "oracle" / o / "batch_*.json")):
        for x in json.load(open(f)):
            items[x["id"]] = len(x["steps"])
    for f in glob.glob(str(EXP / "oracle" / o / "labels_*_*.json")):
        j = Path(f).name.split("_")[1]
        for x in json.load(open(f)):
            labels[x["id"]][j] = int(x["first_error_step"])
agree = {}
for i, v in labels.items():
    spread = max(v.values()) - min(v.values())
    agree[i] = "unanimous" if spread == 0 else "within_1" if spread <= 1 else "within_2" if spread <= 2 else "other"
n_items = len(agree)
cum = [100 * sum(a in ks for a in agree.values()) / n_items
       for ks in (("unanimous",), ("unanimous", "within_1"), ("unanimous", "within_1", "within_2"))]
print(f"\noracle agreement: {n_items} items; unanimous {cum[0]:.0f}%  within±1 {cum[1]:.0f}%  within±2 {cum[2]:.0f}%")
norm = lambda i, j: (labels[i][j] - 1) / max(1, items[i] - 1)
ids = [i for i in labels if i in items and all(labels[i].get(j, 0) > 0 for j, _ in JUDGES)]
fig, ax = plt.subplots(figsize=(8, 5))
parts = ax.violinplot([[norm(i, j) for i in ids] for j, _ in JUDGES], positions=range(3), showmeans=True,
                      showmedians=True, widths=0.7)
for pc, col in zip(parts["bodies"], VIOLIN):
    pc.set_facecolor(col)
    pc.set_alpha(1)
np.random.seed(42)
for cat, col, alpha, z, lab in (("unanimous", "red", 0.4, 5, "Unanimous"), ("within_1", "blue", 0.3, 4, "Within ±1"),
                                ("within_2", "green", 0.25, 3, "Within ±2")):
    sel = [i for i in ids if agree[i] == cat]
    for k, (j, _) in enumerate(JUDGES):
        ax.scatter(k + np.random.normal(0, 0.05, len(sel)), [norm(i, j) for i in sel], c=col, s=8, alpha=alpha,
                   label=lab if k == 0 else "", zorder=z)
ax.set_xticks(range(3))
ax.set_xticklabels([n for _, n in JUDGES], fontsize=14)
ax.set_xlabel("Oracle Model", fontsize=18)
ax.set_ylabel("Normalized Error Position", fontsize=18)
ax.tick_params(axis="y", labelsize=14)
ax.set_ylim(-0.05, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right", fontsize=12)
plt.tight_layout()
save(fig, "oracle_localization_agreement")


# ---- Figs (appendix): chain and thought length distributions (Thought-ICS first attempts) --
import os  # noqa: E402
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402
sys.path.insert(0, str(REPO))
from thought_ics.ics import MODELS as _M  # noqa: E402

chain_len, thought_tok, by_correct = defaultdict(list), defaultdict(list), {True: [], False: []}
for m in FAM_ORDER:
    tok = AutoTokenizer.from_pretrained(_M[m]["hf"])
    for d in sorted(glob.glob(str(EXP / f"{m}_*_oracle_2*/"))):
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



def kde(vals, lim):
    """Gaussian KDE of the values inside the plotted range (a few chains run to hundreds of steps,
    which would otherwise set the bandwidth)."""
    v = np.asarray(vals, float)
    return gaussian_kde(v[v <= lim])


print("\nchain / thought lengths (Thought-ICS first attempts)")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
xc, xt = np.linspace(0, 60, 500), np.linspace(0, 250, 500)
for m in FAM_ORDER:
    cl, tl = chain_len[m], thought_tok[m]
    axes[0].plot(xc, kde(cl, 60)(xc), color=MODEL_COLORS[m], linewidth=2, label=f"{LONG[m]} (median={np.median(cl):.0f})")
    axes[1].plot(xt, kde(tl, 250)(xt), color=MODEL_COLORS[m], linewidth=2, label=f"{LONG[m]} (median={np.median(tl):.0f})")
    print(f"  {m:11s} chains {len(cl):4d}  mean steps {np.mean(cl):5.1f} (median {np.median(cl):3.0f})  "
          f"mean tokens/thought {np.mean(tl):4.0f} (median {np.median(tl):4.0f})")
for ax, xl, title, lim in ((axes[0], "Number of Thought Steps", "Chain Length by Model", 60),
                           (axes[1], "Tokens per Thought", "Thought Length by Model", 250)):
    ax.set_xlabel(xl, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, lim)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
save(fig, "chain_thought_length_by_model")

fig, ax = plt.subplots(figsize=(8, 5))
for k, fill, line, nm in ((True, GREEN, GREEN_DK, "Correct"), (False, RED, RED_DK, "Incorrect")):
    y = kde(by_correct[k], 60)(xc)
    ax.fill_between(xc, y, color=fill, alpha=0.4, label=f"{nm} (median={np.median(by_correct[k]):.0f})")
    ax.plot(xc, y, color=line, linewidth=2.5)
    print(f"  {nm:9s} chains {len(by_correct[k]):5d}  mean steps {np.mean(by_correct[k]):.1f} "
          f"(median {np.median(by_correct[k]):.0f})")
ax.set_xlabel("Number of Thought Steps", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.set_title("Chain Length Distribution: Correct vs Incorrect Traces", fontsize=15)
ax.set_xlim(0, 60)
ax.tick_params(axis="both", labelsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=14)
plt.tight_layout()
save(fig, "chain_length_distribution")


# ---- Fig §5.3: self_correction_lift_errorbar (fully autonomous) ----------------------------
# lift over the shared first attempt (final - first), paired 95% bootstrap CI over problems; a model
# is plotted only when Self-Refine, CoVe and Thought-ICS-F have all finished all six datasets.
def lift_rows():
    out = defaultdict(list)
    for sv in sorted({p.rsplit("_self_2", 1)[0] for p in glob.glob(str(EXP / "*_self_2*"))}):
        lat = lambda pat, f: ([d for d in sorted(glob.glob(pat)) if Path(d, f).exists()] or [None])[-1]
        b = lat(f"{sv}_baselines_*", "cove_results.jsonl")
        fa = lat(f"{sv}_self-fb-{FEEDBACK_VARIANT}_2*", "thought_results.jsonl")
        s = lat(f"{sv}_self_2*", "thought_results.jsonl")
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
done = [m for m in ORDER if len({r["ds"] for r in LR.get(m, [])}) == 6]
print("\nself_correction_lift_errorbar: lift over the first attempt [95% bootstrap CI]")
series = []
for key, name, col in (("sr", "Self-Refine", "darkblue"), ("cove", "CoVe", "#5b9bd5"), ("f", "Thought-ICS-F", "purple")):
    est = []
    for m in ORDER:
        if m not in done:
            est.append((np.nan, np.nan, np.nan))
            continue
        R = LR[m]
        rnd = random.Random(0)
        f = lambda Sx: 100 * sum(r[key] - r["init"] for r in Sx) / len(Sx)
        v = sorted(f([R[rnd.randrange(len(R))] for _ in R]) for _ in range(5000))
        est.append((f(R), v[125], v[4874]))
        print(f"  {name:14s} {m:11s} {est[-1][0]:+5.1f} [{est[-1][1]:+5.1f}, {est[-1][2]:+5.1f}]")
    series.append((name, col, 0.7, est))
if len(done) < len(ORDER):
    print("  not yet complete (omitted):", ", ".join(m for m in ORDER if m not in done))
fig, ax = plt.subplots(figsize=(9, 6))
lift_bars(ax, series, 0.19)
ax.legend(loc="upper left", fontsize=16)
plt.tight_layout()
save(fig, "self_correction_lift_errorbar")
