"""Self-verified (--self-verify, paper L3) report for the shared-init budget study.

    python scripts/budget_study/report_selfverify.py <dir> [--glob '*_selfverify_shared_fbattempt_*']

No oracle in the loop, so the trajectory does not depend on the grader; math datasets are just
regraded with math_verify (as in report.py). Per (model, method):
  - first attempt, final-S (last generated solution), final-A (Thought-ICS-A safeguard: reset to
    the first attempt on V/L disagreement or MaxIter), helped / harmed vs the first attempt;
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
ORDER = ["llama3b", "llama8b", "qwen7b", "qwen14b", "qwen32b", "llama70b", "gptoss20b", "gptoss120b"]
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
ap.add_argument("dir")
ap.add_argument("--glob", default="*_selfverify_shared_fbattempt_*")
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
            finA = i0 if r["exit"] in ("vl_disagreement", "max_iter") else last
            s["n"] += 1
            s["init"] += i0
            s["final_S"] += last
            s["final_A"] += finA
            s["helped_A"] += finA and not i0
            s["harmed_A"] += i0 and not finA
            s["harmed_S"] += i0 and not last
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
                gens = [it for it in r["iterations"] if "gen_status" in it]
                g = lambda it: correct(it["answer"], r["ground_truth"], ds, it["correct"])
                fa = g(gens[0]) if r["exit"] in ("vl_disagreement", "max_iter") else g(gens[-1])
                f.append((fa, g(gens[-1])))
            pairs[m].append((f[0][0], f[1][0], f[0][1], f[1][1]))


def boot(rows, j, B=5000, seed=0):
    rnd = random.Random(seed)
    n = len(rows)
    f = lambda xs: 100 * sum(x[j] - x[j + 1] for x in xs) / len(xs)
    v = sorted(f([rows[rnd.randrange(n)] for _ in range(n)]) for _ in range(B))
    return f(rows), v[int(0.025 * B)], v[int(0.975 * B)]


pct = lambda a, b: f"{100*a/max(1,b):5.1f}"
models = [m for m in ORDER if any((m, x) in stats for x in ("thought", "token"))]
models += sorted({m for m, _ in stats} - set(models))
print("Self-verify accuracy (math regraded with math_verify)")
for m in models:
    for meth in ("thought", "token"):
        s = stats[(m, meth)]
        if not s["n"]:
            continue
        n = s["n"]
        ex = {k[5:]: v for k, v in s.items() if k.startswith("exit_")}
        print(f"  {m:11s}{meth:8s} n={n} init {pct(s['init'],n)}  final-S {pct(s['final_S'],n)}  "
              f"final-A {pct(s['final_A'],n)}  (A: +{s['helped_A']} / -{s['harmed_A']}; S harmed {s['harmed_S']})  exits {ex}")
print("\nThought-ICS minus Token-ICS (paired bootstrap 95% CI)")
for m in models:
    if pairs[m]:
        a, s_ = boot(pairs[m], 0), boot(pairs[m], 2)
        print(f"  {m:11s} final-A {a[0]:+.1f} [{a[1]:+.1f}, {a[2]:+.1f}]   final-S {s_[0]:+.1f} [{s_[1]:+.1f}, {s_[2]:+.1f}]  (n={len(pairs[m])})")
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
