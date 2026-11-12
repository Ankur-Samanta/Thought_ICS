"""Final report for the shared-init budget study (both methods, same first attempts).

    python scripts/budget_study/report.py <full_dir> [--glob '*_shared_fbattempt_*']

Grading: math datasets (amc23, aime, math500) use math_verify equivalence; letter datasets
keep exact-letter grading. Each problem's run is truncated at its first correct attempt under
this grading -- exactly the trajectory a correct grader would have produced (attempts after a
correct answer only existed because the strict grader called it wrong). Strict-grading numbers
are reported alongside.
"""
import argparse
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from math_verify import parse, verify

MATH = {"amc23", "aime", "math500"}
ORDER = ["qwen32b", "gptoss20b", "gptoss120b"]
DS = ["amc23", "aime", "math500", "csqa", "gpqa", "mathqa"]
_cache = {}


def correct(answer, gt, ds, strict):
    if strict or ds not in MATH:
        return strict
    key = (answer, gt)
    if key not in _cache:
        try:
            A, G = parse("$" + answer + "$"), parse("$" + gt + "$")
            _cache[key] = bool(A and G and verify(G, A))
        except Exception:  # noqa: BLE001
            _cache[key] = False
    return _cache[key]


def regrade(rec, ds):
    """Return (init_ok, final_ok, kept_iterations, removed_iterations) under verify grading."""
    its = rec["iterations"]
    for k, it in enumerate(its):
        if "gen_status" not in it:
            continue
        if correct(it["answer"], rec["ground_truth"], ds, it["correct"]):
            return k == 0, True, its[: k + 1], len(its) - (k + 1)
    return False, False, its, 0


ap = argparse.ArgumentParser()
ap.add_argument("full")
ap.add_argument("--glob", default="*_shared_fbattempt_*")
args = ap.parse_args()

P = defaultdict(list)          # model -> list of per-problem tuples
stats = defaultdict(Counter)   # (model, method) -> counters
flips = []
for d in sorted(glob.glob(str(Path(args.full) / (args.glob + "/")))):
    d = Path(d)
    if not (d / "token_results.jsonl").exists():
        continue
    c = json.load(open(d / "config.json"))
    m, ds = c["model"], c["dataset"]
    recs = {meth: {json.loads(l)["problem_id"]: json.loads(l) for l in open(d / f"{meth}_results.jsonl")}
            for meth in ("thought", "token")}
    for pid, t in recs["thought"].items():
        row = {"ds": ds}
        for meth in ("thought", "token"):
            r = recs[meth][pid]
            i0, fin, kept, removed = regrade(r, ds)
            s = stats[(m, meth)]
            s["n"] += 1
            s["init_strict"] += r["iterations"][0]["correct"]
            s["final_strict"] += r["final_correct"]
            s["init"] += i0
            s["final"] += fin
            s["removed_iterations"] += removed
            for it in kept[1:]:
                loc = it.get("localization", {})
                if it.get("failed_localization"):
                    s["failed_loc"] += 1
                elif "gen_status" in it:
                    kind = "from_start" if loc.get("prefix_len", 1) == 0 else "from_prefix"
                    s[kind] += 1
                    s[kind + "_fixed"] += correct(it["answer"], r["ground_truth"], ds, it["correct"])
                elif it.get("stopped"):
                    s["no_error_claimed"] += 1
            if meth == "thought" and i0 and not r["iterations"][0]["correct"]:
                flips.append((m, ds, r["iterations"][0]["answer"], r["ground_truth"]))
            row[meth] = (i0, fin)
        P[m].append(row)


def boot(rows, f, B=5000, seed=0):
    rnd = random.Random(seed)
    n = len(rows)
    est = f(rows)
    v = sorted(f([rows[rnd.randrange(n)] for _ in range(n)]) for _ in range(B))
    return est, v[int(0.025 * B)], v[int(0.975 * B)]


mean = lambda xs: sum(xs) / len(xs)
print("Accuracy (math datasets graded with math_verify; strict grading in brackets)")
for m in ORDER:
    if m not in P:
        continue
    for meth in ("thought", "token"):
        s = stats[(m, meth)]
        n = s["n"]
        print(f"  {m:11s}{meth:8s} n={n} init {100*s['init']/n:5.1f} [{100*s['init_strict']/n:5.1f}]  "
              f"final {100*s['final']/n:5.1f} [{100*s['final_strict']/n:5.1f}]  lift {100*(s['final']-s['init'])/n:+5.1f}  "
              f"| iterations removed by regrade: {s['removed_iterations']}")
print("\nThought-ICS minus Token-ICS (paired bootstrap 95% CI, same first attempts)")
for m in ORDER:
    if m not in P:
        continue
    rows = P[m]
    fin = boot(rows, lambda s: 100 * mean([a["thought"][1] - a["token"][1] for a in s]))
    print(f"  {m:11s} final/lift difference {fin[0]:+.1f} [{fin[1]:+.1f}, {fin[2]:+.1f}]   (n={len(rows)})")
    for ds in DS:
        sub = [a for a in rows if a["ds"] == ds]
        if sub:
            t = sum(a["thought"][1] for a in sub)
            k = sum(a["token"][1] for a in sub)
            i = sum(a["thought"][0] for a in sub)
            print(f"      {ds:8s} n={len(sub):3d}  init {100*i/len(sub):4.0f}  Thought {100*t/len(sub):4.0f}  Token {100*k/len(sub):4.0f}")
print("\nCorrections after regrade (per resample: fixed rate)")
for m in ORDER:
    for meth in ("thought", "token"):
        s = stats[(m, meth)]
        if not s["n"]:
            continue
        loc = s["failed_loc"] + s["from_start"] + s["from_prefix"] + s["no_error_claimed"]
        fr = lambda k: f"{s[k]:4d} ({100*s[k+'_fixed']/max(1,s[k]):4.1f}% fixed)"
        print(f"  {m:11s}{meth:8s} localizations {loc:4d} | failed {s['failed_loc']:4d} ({100*s['failed_loc']/max(1,loc):3.0f}%) "
              f"| from prefix {fr('from_prefix')} | from start {fr('from_start')} | 'no error' {s['no_error_claimed']}")
print(f"\nmath_verify flips (strict-wrong -> correct) on first attempts: {len(flips)}; random sample to spot-check:")
for x in random.Random(1).sample(flips, min(15, len(flips))):
    print("   ", x)
