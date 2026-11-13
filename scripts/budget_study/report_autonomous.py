"""Autonomous setting: Thought-ICS-A / Token-ICS-A (self-verify runs) vs Self-Refine / CoVe,
all from the same first attempts (math datasets regraded with math_verify).

    python scripts/budget_study/report_autonomous.py <full_dir>

Uses, per (model, dataset), the latest *_selfverify_shared_fbattempt_* run with both methods
and the latest *_baselines_* run with both baselines. Paired bootstrap 95% CIs over problems.
"""
import glob
import json
import random
import sys
from collections import defaultdict

from math_verify import parse, verify

MATH = {"amc23", "aime", "math500"}
ORDER = ["llama3b", "qwen7b", "llama8b", "qwen14b", "qwen32b", "llama70b", "gptoss20b", "gptoss120b"]
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


full = sys.argv[1]
rows = defaultdict(list)  # model -> per-problem dict
missing = []
for sv in sorted({d.rsplit("_selfverify", 1)[0] for d in glob.glob(f"{full}/*_selfverify_shared_fbattempt_*")}):
    tag = sv.split("/")[-1]
    s = latest(f"{sv}_selfverify_shared_fbattempt_*", ["thought_results.jsonl", "token_results.jsonl"])
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
        vd = latest(f"{sv}_selfverifyfb{v}_shared_fbattempt_*", ["thought_results.jsonl"])
        VAR[v] = load(f"{vd}/thought_results.jsonl") if vd else None
    for pid, t in T.items():
        gt = t["ground_truth"]
        g = lambda it: ok(it["answer"], gt, ds, it["correct"])

        def final_a(r):
            gens = [it for it in r["iterations"] if "gen_status" in it]
            return g(gens[0]) if r["exit"] in ("vl_disagreement", "max_iter") else g(gens[-1])
        def final_s(r):
            return g([it for it in r["iterations"] if "gen_status" in it][-1])
        row = {"ds": ds, "init": g(t["iterations"][0]), "thought": final_a(t), "token": final_a(K[pid]),
               "thought_s": final_s(t), "self_refine": g(SR[pid]["iterations"][-1]), "cove": g(CV[pid]["iterations"][-1])}
        for v in ("loc", "all"):
            if VAR[v] is not None:
                row[f"fb{v}"] = final_s(VAR[v][pid])
        rows[m].append(row)


def boot(xs, a, b, B=5000, seed=0):
    rnd = random.Random(seed)
    f = lambda s: 100 * sum(x[a] - x[b] for x in s) / len(s)
    v = sorted(f([xs[rnd.randrange(len(xs))] for _ in xs]) for _ in range(B))
    return f(xs), v[int(.025 * B)], v[int(.975 * B)]


fmt = lambda t: f"{t[0]:+5.1f} [{t[1]:+5.1f}, {t[2]:+5.1f}]"
print(f"{'model':11s}{'n':>5s} {'ds':>3s} | {'first':>5s} {'Th-A':>5s} {'Tok-A':>5s} {'SR':>5s} {'CoVe':>5s} | "
      f"{'Th-A - SR':>22s} {'Th-A - CoVe':>22s} {'Th-A - Tok-A':>22s}")
for m in [x for x in ORDER if x in rows] + sorted(set(rows) - set(ORDER)):
    R = rows[m]
    p = lambda k: 100 * sum(r[k] for r in R) / len(R)
    print(f"{m:11s}{len(R):5d} {len({r['ds'] for r in R}):3d} | {p('init'):5.1f} {p('thought'):5.1f} {p('token'):5.1f} "
          f"{p('self_refine'):5.1f} {p('cove'):5.1f} | {fmt(boot(R, 'thought', 'self_refine')):>22s} "
          f"{fmt(boot(R, 'thought', 'cove')):>22s} {fmt(boot(R, 'thought', 'token')):>22s}")
if missing:
    print("\nnot finished:", ", ".join(missing))

print("\nFeedback variants (Thought-ICS, last solution; only datasets where the variant finished)")
print(f"{'model':11s}{'var':5s}{'n':>5s} {'ds':>3s} | {'first':>5s} {'Th-S':>5s} {'SR':>5s} {'CoVe':>5s} {'Th-fb':>5s} | "
      f"{'Th-fb - SR':>22s} {'Th-fb - Th-S':>22s} {'Th-fb - CoVe':>22s}")
for m in [x for x in ORDER if x in rows]:
    for v in ("loc", "all"):
        R = [r for r in rows[m] if f"fb{v}" in r]
        if not R:
            continue
        p = lambda k: 100 * sum(r[k] for r in R) / len(R)
        print(f"{m:11s}{v:5s}{len(R):5d} {len({r['ds'] for r in R}):3d} | {p('init'):5.1f} {p('thought_s'):5.1f} {p('self_refine'):5.1f} "
              f"{p('cove'):5.1f} {p(f'fb{v}'):5.1f} | {fmt(boot(R, f'fb{v}', 'self_refine')):>22s} "
              f"{fmt(boot(R, f'fb{v}', 'thought_s')):>22s} {fmt(boot(R, f'fb{v}', 'cove')):>22s}")
