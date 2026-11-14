"""Fully autonomous setting: Thought-ICS-F vs Thought-ICS / Token-ICS without feedback, Self-Refine
and CoVe, all from the same first attempts (math benchmarks regraded with Math-Verify).

    python scripts/report_autonomous.py [results_dir]

Per (model, benchmark) it uses the latest finished run of each: *_self_2* (Thought-ICS and Token-ICS
without feedback), *_self-fb-all_2* (Thought-ICS-F), *_self-fb-loc_2* (ablation), *_baselines_2*.
Accuracy is the last generated solution. Paired bootstrap 95% CIs over problems.
"""
import glob
import json
import random
import sys
from collections import defaultdict

from math_verify import parse, verify

MATH = {"amc23", "aime", "math500"}
ORDER = ["llama3b", "qwen7b", "llama8b", "qwen14b", "gptoss20b", "qwen32b", "llama70b", "gptoss120b"]
_c = {}


def ok(ans, gt, ds, strict):
    if strict or ds not in MATH:
        return strict
    if (ans, gt) not in _c:
        try:
            A, G = parse("$" + ans + "$"), parse("$" + gt + "$")
            _c[(ans, gt)] = bool(A and G and verify(G, A))
        except Exception:  # noqa: BLE001
            _c[(ans, gt)] = False
    return _c[(ans, gt)]


def latest(pat, need):
    ds = [d for d in sorted(glob.glob(pat)) if all(glob.glob(f"{d}/{n}") for n in need)]
    return ds[-1] if ds else None


full = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = defaultdict(list)  # model -> per-problem dict
missing = []
for sv in sorted({d.rsplit("_self_2", 1)[0] for d in glob.glob(f"{full}/*_self_2*")}):
    tag = sv.split("/")[-1]
    s = latest(f"{sv}_self_2*", ["thought_results.jsonl", "token_results.jsonl"])
    b = latest(f"{sv}_baselines_*", ["self_refine_results.jsonl", "cove_results.jsonl"])
    if not s or not b:
        missing.append(tag)
        continue
    c = json.load(open(f"{s}/config.json"))
    m, ds = c["model"], c["dataset"]
    load = lambda p: {json.loads(l)["problem_id"]: json.loads(l) for l in open(p)}
    T, K = load(f"{s}/thought_results.jsonl"), load(f"{s}/token_results.jsonl")
    SR, CV = load(f"{b}/self_refine_results.jsonl"), load(f"{b}/cove_results.jsonl")
    VAR = {}
    for v in ("loc", "all"):
        vd = latest(f"{sv}_self-fb-{v}_2*", ["thought_results.jsonl"])
        VAR[v] = load(f"{vd}/thought_results.jsonl") if vd else None
    for pid, t in T.items():
        gt = t["ground_truth"]
        g = lambda it: ok(it["answer"], gt, ds, it["correct"])

        def final(r):
            return g([it for it in r["iterations"] if "gen_status" in it][-1])
        row = {"ds": ds, "init": g(t["iterations"][0]), "thought": final(t), "token": final(K[pid]),
               "self_refine": g(SR[pid]["iterations"][-1]), "cove": g(CV[pid]["iterations"][-1])}
        for v in ("loc", "all"):
            if VAR[v] is not None:
                row[f"fb{v}"] = final(VAR[v][pid])
        rows[m].append(row)


def boot(xs, a, b, B=5000, seed=0):
    rnd = random.Random(seed)
    f = lambda s: 100 * sum(x[a] - x[b] for x in s) / len(s)
    v = sorted(f([xs[rnd.randrange(len(xs))] for _ in xs]) for _ in range(B))
    return f(xs), v[int(.025 * B)], v[int(.975 * B)]


fmt = lambda t: f"{t[0]:+5.1f} [{t[1]:+5.1f}, {t[2]:+5.1f}]"
models = [x for x in ORDER if x in rows] + sorted(set(rows) - set(ORDER))
print("Fully autonomous self-correction (accuracy %, same first attempts)")
print(f"{'model':11s}{'n':>5s} {'ds':>3s} | {'first':>5s} {'Th':>5s} {'Tok':>5s} {'SR':>5s} {'CoVe':>5s} {'Th-F':>5s} | "
      f"{'Th-F - SR':>22s} {'Th-F - CoVe':>22s} {'Th-F - Th':>22s}")
for m in models:
    R = [r for r in rows[m] if "fball" in r]
    if not R:
        continue
    p = lambda k: 100 * sum(r[k] for r in R) / len(R)
    print(f"{m:11s}{len(R):5d} {len({r['ds'] for r in R}):3d} | {p('init'):5.1f} {p('thought'):5.1f} {p('token'):5.1f} "
          f"{p('self_refine'):5.1f} {p('cove'):5.1f} {p('fball'):5.1f} | {fmt(boot(R, 'fball', 'self_refine')):>22s} "
          f"{fmt(boot(R, 'fball', 'cove')):>22s} {fmt(boot(R, 'fball', 'thought')):>22s}")
print("\nAblation: verification feedback to the localizer only (Thought-ICS)")
print(f"{'model':11s}{'n':>5s} {'ds':>3s} | {'Th':>5s} {'loc':>5s} {'all':>5s} | {'loc - Th':>22s} {'all - Th':>22s}")
for m in models:
    R = [r for r in rows[m] if "fbloc" in r and "fball" in r]
    if not R:
        continue
    p = lambda k: 100 * sum(r[k] for r in R) / len(R)
    print(f"{m:11s}{len(R):5d} {len({r['ds'] for r in R}):3d} | {p('thought'):5.1f} {p('fbloc'):5.1f} {p('fball'):5.1f} | "
          f"{fmt(boot(R, 'fbloc', 'thought')):>22s} {fmt(boot(R, 'fball', 'thought')):>22s}")
if missing:
    print("\nnot finished:", ", ".join(missing))
