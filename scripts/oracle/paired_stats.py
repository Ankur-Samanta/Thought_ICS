"""Paired oracle-localization statistics across oracle runs (rows.json from analyze.py).

    python scripts/oracle/paired_stats.py <oracle_dir> [<oracle_dir> ...]

Per model (paired on the same wrong first attempt, Thought-ICS vs Token-ICS):
  clean  = localized at or before the consensus first error (failed localization = not clean)
  exact  = localized exactly at the consensus first error (failed = not exact)
  usable = clean among non-failed localizations only (quality given a usable answer)
Fix rate after a clean vs an erroneous ("later") prefix, pooled per method and model group.
95% CIs: paired bootstrap over items (5000 resamples, seed 0).
"""
import json
import random
import sys
from collections import defaultdict

ORDER = ["llama3b", "qwen7b", "llama8b", "qwen14b", "qwen32b", "llama70b", "gptoss20b", "gptoss120b"]
rows = [r for d in sys.argv[1:] for r in json.load(open(f"{d}/rows.json"))]
by = defaultdict(dict)
for r in rows:
    by[(r["model"], r["id"])][r["method"]] = r
pairs = defaultdict(list)
for (m, _), v in by.items():
    if "thought" in v and "token" in v:
        pairs[m].append((v["thought"], v["token"]))

clean = lambda r: r["cat"] in ("exact", "earlier")
exact = lambda r: r["cat"] == "exact"


def boot(xs, f, B=5000, seed=0):
    rnd = random.Random(seed)
    n = len(xs)
    est = f(xs)
    v = sorted(f([xs[rnd.randrange(n)] for _ in range(n)]) for _ in range(B))
    return est, v[int(.025 * B)], v[int(.975 * B)]


def pct(xs, g):
    return 100 * sum(g(x) for x in xs) / max(1, len(xs))


def diff(g):
    return lambda P: 100 * sum(g(a) - g(b) for a, b in P) / len(P)


fmt = lambda t: f"{t[0]:+5.1f} [{t[1]:+5.1f}, {t[2]:+5.1f}]"
models = [m for m in ORDER if m in pairs] + sorted(set(pairs) - set(ORDER))
print(f"{'model':11s}{'n':>5s} | {'clean T / K':>12s}  {'T-K [95% CI]':>22s} | {'exact T / K':>12s}  {'T-K [95% CI]':>22s} | "
      f"{'failed T / K':>12s} | {'clean|usable T / K':>18s}")
for m in models:
    P = pairs[m]
    T, K = [a for a, _ in P], [b for _, b in P]
    usable = lambda xs: [x for x in xs if x["cat"] != "failed"]
    print(f"{m:11s}{len(P):5d} | {pct(T, clean):5.0f} / {pct(K, clean):3.0f}  {fmt(boot(P, diff(clean))):>22s} | "
          f"{pct(T, exact):5.0f} / {pct(K, exact):3.0f}  {fmt(boot(P, diff(exact))):>22s} | "
          f"{pct(T, lambda x: x['cat'] == 'failed'):5.0f} / {pct(K, lambda x: x['cat'] == 'failed'):3.0f} | "
          f"{pct(usable(T), clean):8.0f} / {pct(usable(K), clean):3.0f}")
groups = {"small (3B-14B)": ["llama3b", "qwen7b", "llama8b", "qwen14b"],
          "large (20B-120B)": ["qwen32b", "llama70b", "gptoss20b", "gptoss120b"], "all": models}
for g, ms in groups.items():
    P = [p for m in ms for p in pairs.get(m, [])]
    if P:
        print(f"{g:17s} n={len(P):4d}  clean T-K {fmt(boot(P, diff(clean)))}   exact T-K {fmt(boot(P, diff(exact)))}")

print("\nFix rate of the first resample: after a clean prefix vs after an erroneous (later) prefix")
for g, ms in groups.items():
    for meth in ("thought", "token"):
        R = [r for r in rows if r["model"] in ms and r["method"] == meth and r["cat"] != "failed"]
        C = [r["fixed"] for r in R if clean(r)]
        E = [r["fixed"] for r in R if r["cat"] == "later"]
        if not C or not E:
            continue
        rnd = random.Random(0)
        d = sorted(100 * (sum(rnd.choice(C) for _ in C) / len(C) - sum(rnd.choice(E) for _ in E) / len(E))
                   for _ in range(5000))
        est = 100 * (sum(C) / len(C) - sum(E) / len(E))
        print(f"  {g:17s}{meth:8s} clean {100*sum(C)/len(C):5.1f}% (n={len(C):4d})  erroneous {100*sum(E)/len(E):5.1f}% "
              f"(n={len(E):4d})  diff {est:+5.1f} [{d[125]:+5.1f}, {d[4875]:+5.1f}]")
