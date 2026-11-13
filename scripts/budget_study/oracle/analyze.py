"""Clean vs erroneous prefixes against an oracle consensus, for both methods.

    python scripts/budget_study/oracle_analyze.py <full_dir> <oracle_dir>

Reads labels_<judge>_<k>.json from <oracle_dir>. Consensus per item: a step chosen by >= 2
judges (exact), else the median of the judges' steps; items whose consensus is 0 (no error
found) are dropped. Each method's FIRST localization (iteration 1, on the shared first attempt)
is classified as exact / earlier (both = clean prefix) / later (erroneous prefix) / failed
(unusable localization, Token-ICS only in practice). A Token-ICS cut position maps to the
step that contains it (the Token-ICS solution is the joined Thought-ICS chain).
"""
import glob
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

full, odir = Path(sys.argv[1]), Path(sys.argv[2])
votes = defaultdict(dict)
for f in glob.glob(str(odir / "labels_*_*.json")):
    judge = Path(f).name.split("_")[1]
    for x in json.load(open(f)):
        votes[x["id"]][judge] = int(x["first_error_step"])

judges = sorted({j for v in votes.values() for j in v})
agree = Counter()
consensus = {}
for i, v in votes.items():
    steps = list(v.values())
    c = Counter(steps).most_common(1)[0]
    consensus[i] = c[0] if c[1] >= 2 else int(st.median(steps))
    agree["unanimous"] += len(set(steps)) == 1
    agree["within1"] += max(steps) - min(steps) <= 1
    agree["n"] += 1
print(f"judges={judges} items={agree['n']} unanimous={agree['unanimous']} within±1={agree['within1']}")

runs = {}
for d in glob.glob(str(full / "*_L2_shared_fbattempt_*/")):
    c = json.load(open(Path(d) / "config.json"))
    runs[(c["model"], c["dataset"])] = Path(d)

R = defaultdict(lambda: defaultdict(list))
ROWS = []  # per item x method, written to <oracle_dir>/rows.json for paired statistics
missing = set()
for i, k in consensus.items():
    if k == 0:
        continue
    model, ds, pid = i.split("|")
    d = runs[(model, ds)]
    for meth in ("thought", "token"):
        if not (d / f"{meth}_results.jsonl").exists():
            missing.add(f"{model}/{ds}/{meth}")
            continue
        rec = next(json.loads(l) for l in open(d / f"{meth}_results.jsonl") if json.loads(l)["problem_id"] == pid)
        its = rec["iterations"]
        if len(its) < 2 or "localization" not in its[1]:
            continue
        it1, loc = its[1], its[1]["localization"]
        if it1.get("failed_localization"):
            R[(model, meth)]["failed"].append(None)
            ROWS.append({"id": i, "model": model, "dataset": ds, "method": meth, "cat": "failed", "step": None,
                         "oracle": k, "n_steps": None, "fixed": None})
            continue
        if "gen_status" not in it1:
            continue  # localizer said no error
        if meth == "thought":
            s = loc["error_step"]
        else:
            thought0 = next(json.loads(l) for l in open(d / "thought_results.jsonl")
                            if json.loads(l)["problem_id"] == pid)["iterations"][0]["chain"]
            idx, pos, s = loc["truncation_idx"] or 0, 0, 1
            for j, t in enumerate(thought0, 1):
                if idx < pos + len(t):
                    s = j
                    break
                pos += len(t)
            else:
                s = len(thought0)
        cat = "exact" if s == k else ("earlier" if s < k else "later")
        R[(model, meth)][cat].append(it1["correct"])
        ROWS.append({"id": i, "model": model, "dataset": ds, "method": meth, "cat": cat, "step": s,
                     "oracle": k, "n_steps": len(its[0].get("chain") or []) or None, "fixed": it1["correct"],
                     "answer": it1.get("answer"), "ground_truth": rec["ground_truth"]})

print(f"\n{'model':11s}{'method':8s}{'n':>4s} | {'exact':>6s}{'earlier':>8s}{'later':>7s}{'failed':>7s} | {'clean':>6s} | fix rate: clean vs erroneous")
for (model, meth), c in sorted(R.items()):
    n = sum(len(v) for v in c.values())
    clean = c["exact"] + c["earlier"]
    pct = lambda k: 100 * len(c[k]) / max(1, n)
    fr = lambda xs: f"{100 * sum(xs) / len(xs):4.0f}% (n={len(xs)})" if xs else "   -"
    print(f"{model:11s}{meth:8s}{n:4d} | {pct('exact'):5.0f}%{pct('earlier'):7.0f}%{pct('later'):6.0f}%{pct('failed'):6.0f}% | "
          f"{100 * len(clean) / max(1, n):5.0f}% | {fr(clean)} vs {fr(c['later'])}")
if missing:
    print("\nnot finished yet (skipped):", ", ".join(sorted(missing)))
(odir / "rows.json").write_text(json.dumps(ROWS))
