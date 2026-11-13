"""Appendix tables for the budget-study rerun, written as LaTeX snippets.

    python scripts/budget_study/figures/make_tables.py [out_dir]

Default out_dir: TREE/thoughtics_paper/paper_v2/tables. Writes:
  l2_results.tex   self-correction under oracle verification, per model x dataset:
                   Init (the first attempt shared by both methods), Token-ICS, Thought-ICS final accuracy.
Math datasets regraded with math_verify, runs truncated at the first correct attempt (as report.py).
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

from math_verify import parse, verify

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments_budget_study"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/asamanta1/SIERA/TREE/thoughtics_paper/paper_v2/tables")
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["llama3b", "qwen7b", "llama8b", "qwen14b", "gptoss20b", "qwen32b", "llama70b", "gptoss120b"]
NAME = {"llama3b": "LLaMA-3B", "qwen7b": "Qwen-7B", "llama8b": "LLaMA-8B", "qwen14b": "Qwen-14B",
        "gptoss20b": "OSS-20B", "qwen32b": "Qwen-32B", "llama70b": "LLaMA-70B", "gptoss120b": "OSS-120B"}
DS = [("amc23", "AMC23"), ("aime", "AIME"), ("math500", "MATH-L5"), ("csqa", "CSQA"), ("gpqa", "GPQA")]
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


def first_correct(rec, ds):
    for k, it in enumerate(rec["iterations"]):
        if "gen_status" in it and ok(it["answer"], rec["ground_truth"], ds, it["correct"]):
            return k == 0, True
    return False, False


acc = defaultdict(dict)  # (model, ds) -> {"init","token","thought"}
for d in sorted(glob.glob(str(EXP / "full" / "*_L2_shared_fbattempt_*/"))):
    d = Path(d)
    if not (d / "token_results.jsonl").exists():
        continue
    c = json.load(open(d / "config.json"))
    load = lambda m: [json.loads(l) for l in open(d / f"{m}_results.jsonl")]
    T, K = load("thought"), load("token")
    n = len(T)
    it = [first_correct(r, c["dataset"]) for r in T]
    tk = [first_correct(r, c["dataset"]) for r in K]
    acc[(c["model"], c["dataset"])] = {"init": 100 * sum(a for a, _ in it) / n, "thought": 100 * sum(b for _, b in it) / n,
                                       "token": 100 * sum(b for _, b in tk) / n}

# MV@10: latest *_mv10_* run per (model, dataset); voted answer regraded with math_verify
for d in sorted(glob.glob(str(EXP / "full" / "*_mv10_*/"))):
    d = Path(d)
    if not (d / "mv_results.jsonl").exists():
        continue
    c = json.load(open(d / "config.json"))
    R = [json.loads(l) for l in open(d / "mv_results.jsonl")]
    if (c["model"], c["dataset"]) in acc:
        acc[(c["model"], c["dataset"])]["mv"] = 100 * sum(ok(r["voted"], r["ground_truth"], c["dataset"], r["correct"])
                                                           for r in R) / len(R)
HAVE_MV = all("mv" in acc[(m, ds)] for m in ORDER for ds, _ in DS)
print("MV@10 rows:", "included" if HAVE_MV else "not all runs finished -- omitted")

lines = []
for i, m in enumerate(ORDER):
    rows = [("Init", "init"), ("Token-ICS", "token")] + ([("MV@10", "mv")] if HAVE_MV else []) + [("Thought-ICS", "thought")]
    for j, (label, key) in enumerate(rows):
        cells = []
        for ds, _ in DS:
            a = acc[(m, ds)]
            best = max(a[k] for _, k in rows)
            v = a[key]
            cells.append(f"\\textbf{{{v:.1f}}}" if abs(v - best) < 1e-9 else f"{v:.1f}")
        lines.append(f"{NAME[m] if j == 0 else ''} & {label} & " + " & ".join(cells) + " \\\\")
    if i < len(ORDER) - 1:
        lines.append("\\midrule")
(OUT / "l2_results.tex").write_text("\n".join(lines) + "\n")
print("wrote", OUT / "l2_results.tex")
print("\n".join(lines[:4]))


# ---- localization_paired.tex: Thought-ICS vs Token-ICS localization on the same responses ----
import random  # noqa: E402

rows = [r for o in ("oracle_full", "oracle_slm") for r in json.load(open(EXP / o / "rows.json"))]
pair = defaultdict(dict)
for r in rows:
    pair[r["id"]][r["method"]] = r
P = [v for v in pair.values() if "thought" in v and "token" in v]


def boot_diff(S, g, B=5000, seed=0):
    rnd = random.Random(seed)
    f = lambda xs: 100 * sum(g(a) - g(b) for a, b in xs) / len(xs)
    v = sorted(f([S[rnd.randrange(len(S))] for _ in S]) for _ in range(B))
    return f(S), v[int(.025 * B)], v[int(.975 * B)]


later = lambda r: r["cat"] == "later"
failed = lambda r: r["cat"] == "failed"
out = []
groups = [(NAME[m], [m]) for m in ORDER] + [("14B and larger", ["qwen14b", "gptoss20b", "qwen32b", "llama70b", "gptoss120b"])]
for gi, (label, ms) in enumerate(groups):
    A = [(v["thought"], v["token"]) for v in P if v["thought"]["model"] in ms]
    S = [(a, b) for a, b in A if a["cat"] != "failed" and b["cat"] != "failed"]
    lt, lk = (100 * sum(later(x[i]) for x in S) / len(S) for i in (0, 1))
    e, lo, hi = boot_diff(S, later)
    ft, fk = (100 * sum(failed(x[i]) for x in A) / len(A) for i in (0, 1))
    if gi == len(groups) - 1:
        out.append("\\midrule")
    out.append(f"{label} & {len(S)} & {lt:.0f} / {lk:.0f} & ${e:+.1f}$ [${lo:+.1f}$, ${hi:+.1f}$] & {ft:.0f} / {fk:.0f} \\\\")
(OUT / "localization_paired.tex").write_text("\n".join(out) + "\n")
print("wrote", OUT / "localization_paired.tex")
print("\n".join(out))
