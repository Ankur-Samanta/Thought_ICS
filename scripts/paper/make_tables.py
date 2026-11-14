"""Appendix tables, written as LaTeX snippets.

    python scripts/paper/make_tables.py [out_dir]

Default out_dir: paper_assets/tables. Writes:
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

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper_assets" / "tables"
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
for d in sorted(glob.glob(str(EXP / "*_oracle_2*/"))):
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
for d in sorted(glob.glob(str(EXP / "*_mv10_*/"))):
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

rows = [r for o in ("part1", "part2") for r in json.load(open(EXP / "oracle" / o / "rows.json"))]
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


# ---- compute_analysis.tex and correction_diversity.tex (oracle-verified runs) --------------
def truncated(rec, ds):
    """Iterations up to and including the first correct generation (math_verify), as report.py."""
    its = rec["iterations"]
    for k, it in enumerate(its):
        if "gen_status" in it and ok(it["answer"], rec["ground_truth"], ds, it["correct"]):
            return its[:k + 1], True
    return its, False


comp = defaultdict(lambda: defaultdict(lambda: {"new": 0, "kept": 0, "resampled": 0, "n": 0, "ok": 0}))
div = defaultdict(lambda: defaultdict(list))
for d in sorted(glob.glob(str(EXP / "*_oracle_2*/"))):
    d = Path(d)
    if not (d / "token_results.jsonl").exists():
        continue
    c = json.load(open(d / "config.json"))
    for meth in ("token", "thought"):
        C = comp[c["model"]][meth]
        for line in open(d / f"{meth}_results.jsonl"):
            r = json.loads(line)
            its, fin = truncated(r, c["dataset"])
            C["n"] += 1
            C["ok"] += fin
            gens = []
            for it in its:
                loc = it.get("localization")
                if loc:
                    C["new"] += loc.get("n_tokens", 0)
                if "gen_status" in it:
                    gens.append((it["answer"] or "").strip())
                    if it["iteration"] > 0:
                        C["new"] += it["n_new_tokens"]
                        C["kept"] += it["n_tokens"] - it["n_new_tokens"]
                        C["resampled"] += it["n_tokens"]
            if len(gens) - 1 > 1:  # more than one correction iteration
                div[c["model"]][meth].append(gens)

# Thought-ICS only (descriptive): initial tokens, correction tokens, correction iterations, saved, accuracy
ini = defaultdict(lambda: {"tok": 0, "iters": 0, "n": 0})
for d in sorted(glob.glob(str(EXP / "*_oracle_2*/"))):
    d = Path(d)
    if not (d / "token_results.jsonl").exists():
        continue
    c = json.load(open(d / "config.json"))
    for line in open(d / "thought_results.jsonl"):
        r = json.loads(line)
        its, _ = truncated(r, c["dataset"])
        I = ini[c["model"]]
        I["tok"] += its[0]["n_tokens"]
        I["iters"] += sum(1 for it in its[1:] if "localization" in it)
        I["n"] += 1
lines = []
T = defaultdict(float)
for m in ORDER:
    C, I = comp[m]["thought"], ini[m]
    row = (I["tok"] / I["n"], C["new"] / C["n"], I["iters"] / I["n"], 100 * C["kept"] / max(1, C["resampled"]), 100 * C["ok"] / C["n"])
    for k, v in zip(("i", "n", "it", "s", "a"), row):
        T[k] += v / len(ORDER)
    f = lambda v: f"{v:,.0f}".replace(",", "{,}")
    lines.append(f"{NAME[m]} & {f(row[0])} & {f(row[1])} & {row[2]:.1f} & {row[3]:.0f}\\% & {row[4]:.1f} \\\\")
lines.append("\\midrule")
f = lambda v: f"{v:,.0f}".replace(",", "{,}")
lines.append(f"\\textbf{{Average}} & {f(T['i'])} & {f(T['n'])} & {T['it']:.1f} & {T['s']:.0f}\\% & {T['a']:.1f} \\\\")
(OUT / "compute_analysis.tex").write_text("\n".join(lines) + "\n")
print("\nwrote compute_analysis.tex"); print("\n".join(lines))

lines = []
agg = {m: [] for m in ("token", "thought")}
for m in ORDER:
    for j, meth in enumerate(("token", "thought")):
        G = div[m][meth]
        agg[meth] += G
        uniq = sum(len(set(g)) / len(g) for g in G) / max(1, len(G))
        pairs = [(a, b) for g in G for a, b in zip(g, g[1:])]
        same = 100 * sum(a == b for a, b in pairs) / max(1, len(pairs))
        ch = [x != g[0] for g in G for x in g[1:]]
        changed = 100 * sum(ch) / max(1, len(ch))
        iters = sum(len(g) for g in G) / max(1, len(G))
        first = f"\\multirow{{2}}{{*}}{{{NAME[m]}}}" if j == 0 else ""
        lines.append(f"{first:28s} & {'Token-ICS' if meth == 'token' else 'Thought-ICS'} & {len(G)} & {iters:.1f} & {uniq:.2f} & {same:.1f}\\% & {changed:.1f}\\% \\\\")
lines.append("\\midrule")
for j, meth in enumerate(("token", "thought")):
    G = agg[meth]
    uniq = sum(len(set(g)) / len(g) for g in G) / len(G)
    pairs = [(a, b) for g in G for a, b in zip(g, g[1:])]
    same = 100 * sum(a == b for a, b in pairs) / len(pairs)
    ch = [x != g[0] for g in G for x in g[1:]]
    first = "\\multirow{2}{*}{\\textbf{Average}}" if j == 0 else ""
    lines.append(f"{first:28s} & {'Token-ICS' if meth == 'token' else 'Thought-ICS'} & {len(G):,} & {sum(len(g) for g in G)/len(G):.1f} & {uniq:.2f} & {same:.1f}\\% & {100*sum(ch)/len(ch):.1f}\\% \\\\".replace(",", "{,}"))
(OUT / "correction_diversity.tex").write_text("\n".join(lines) + "\n")
print("\nwrote correction_diversity.tex"); print("\n".join(lines))
