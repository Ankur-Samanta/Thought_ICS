"""Self-verified runs: accuracy, exits and verifier quality.

    python scripts/report_self_verified.py [results_dir] [--glob '*_self_2*']

No oracle in the loop, so the trajectory does not depend on the grader; math benchmarks are regraded
with Math-Verify. Per (model, method):
  - first attempt, final (the last generated solution), fixed / broken relative to the first attempt;
  - exit reasons;
  - verifier: unparseable rate, budget hits, median tokens, recall (says NO on a wrong solution)
    and specificity (says YES on a correct one), graded against the solution it was shown.
"""
import argparse
import glob
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from math_verify import parse, verify

MATH = {"amc23", "aime", "math500"}
ORDER = ["llama3b", "qwen7b", "llama8b", "qwen14b", "gptoss20b", "qwen32b", "llama70b", "gptoss120b"]
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


ap = argparse.ArgumentParser()
ap.add_argument("dir", nargs="?", default="results")
ap.add_argument("--glob", default="*_self_2*")
args = ap.parse_args()

stats = defaultdict(Counter)
vtok = defaultdict(list)
pairs = defaultdict(list)  # model -> (thought_final_A, token_final_A, thought_final_S, token_final_S)
for d in sorted(glob.glob(str(Path(args.dir) / (args.glob + "/")))):
    d = Path(d)
    c = json.load(open(d / "config.json"))
    m, ds = c["model"], c["dataset"]
    recs = {}
    for meth in ("thought", "token"):
        p = d / f"{meth}_results.jsonl"
        if p.exists():
            recs[meth] = {json.loads(l)["problem_id"]: json.loads(l) for l in open(p)}
    for meth, R in recs.items():
        s = stats[(m, meth)]
        for r in R.values():
            gt = r["ground_truth"]
            its = r["iterations"]
            ok = lambda it: correct(it["answer"], gt, ds, it["correct"])
            gens = [it for it in its if "gen_status" in it]
            i0, last = ok(gens[0]), ok(gens[-1])
            s["n"] += 1
            s["init"] += i0
            s["final"] += last
            s["fixed"] += last and not i0
            s["broken"] += i0 and not last
            s["exit_" + r["exit"]] += 1
            cur = i0
            for it in its[1:]:
                v = it.get("verification") or it.get("localization", {}).get("verification")
                if v is not None:
                    s["verif"] += 1
                    s["v_unparse"] += v["verdict"] is None
                    s["v_cap"] += v["finish_reason"] == "length"
                    vtok[(m, meth)].append(v["n_tokens"])
                    says_ok = v["verdict"] is True
                    s["v_on_correct" if cur else "v_on_wrong"] += 1
                    s["v_yes_on_correct" if cur else "v_no_on_wrong"] += (says_ok if cur else not says_ok)
                if "gen_status" in it:
                    cur = ok(it)
    if set(recs) == {"thought", "token"}:
        for pid, t in recs["thought"].items():
            k = recs["token"][pid]
            f = []
            for r in (t, k):
                last = [it for it in r["iterations"] if "gen_status" in it][-1]
                f.append(correct(last["answer"], r["ground_truth"], ds, last["correct"]))
            pairs[m].append((f[0], f[1]))


def boot(rows, j, B=5000, seed=0):
    rnd = random.Random(seed)
    n = len(rows)
    f = lambda xs: 100 * sum(x[j] - x[j + 1] for x in xs) / len(xs)
    v = sorted(f([rows[rnd.randrange(n)] for _ in range(n)]) for _ in range(B))
    return f(rows), v[int(0.025 * B)], v[int(0.975 * B)]


pct = lambda a, b: f"{100*a/max(1,b):5.1f}"
models = [m for m in ORDER if any((m, x) in stats for x in ("thought", "token"))]
models += sorted({m for m, _ in stats} - set(models))
print("Self-verified accuracy (math regraded with Math-Verify)")
for m in models:
    for meth in ("thought", "token"):
        s = stats[(m, meth)]
        if not s["n"]:
            continue
        n = s["n"]
        ex = {k[5:]: v for k, v in s.items() if k.startswith("exit_")}
        print(f"  {m:11s}{meth:8s} n={n} init {pct(s['init'],n)}  final {pct(s['final'],n)}  "
              f"(fixed {s['fixed']}, broken {s['broken']})  exits {ex}")
print("\nThought-ICS minus Token-ICS (paired bootstrap 95% CI)")
for m in models:
    if pairs[m]:
        a = boot(pairs[m], 0)
        print(f"  {m:11s} final {a[0]:+.1f} [{a[1]:+.1f}, {a[2]:+.1f}]  (n={len(pairs[m])})")
print("\nVerifier")
for m in models:
    for meth in ("thought", "token"):
        s = stats[(m, meth)]
        if not s["verif"]:
            continue
        t = vtok[(m, meth)]
        print(f"  {m:11s}{meth:8s} calls {s['verif']:4d}  unparseable {pct(s['v_unparse'],s['verif'])}%  "
              f"budget hits {pct(s['v_cap'],s['verif'])}%  median tok {statistics.median(t):6.0f}  "
              f"recall(NO|wrong) {pct(s['v_no_on_wrong'],s['v_on_wrong'])}% (n={s['v_on_wrong']})  "
              f"specificity(YES|correct) {pct(s['v_yes_on_correct'],s['v_on_correct'])}% (n={s['v_on_correct']})")
