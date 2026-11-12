"""Judge thought atomicity for step-prompt variants produced by diag_step_prompts.py.

    python scripts/budget_study/judge_variants.py <variants_dir> <out.json>
"""
import glob
import json
import sys
from collections import Counter, defaultdict

from thought_ics.budget_study import run_study as R
from thought_ics.budget_study.judge import judge

root, out_path = sys.argv[1], sys.argv[2]
MAX_THOUGHTS_JUDGED = 12
items, gen = [], defaultdict(list)
for f in sorted(glob.glob(f"{root}/*/*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        gen[(r["variant"], r["model"])].append(r)
        ch = [t for t in r["chain"] if t.strip()]
        for k, step in enumerate(ch[:MAX_THOUGHTS_JUDGED]):
            items.append({"variant": r["variant"], "model": r["model"], "question": r["question"],
                          "history": ch[:k], "step": step})
m = R.MODELS["gptoss120b"]
eng = R.Engine(m["hf"], m["tp"], m["max_model_len"], 42, 0.85, m["final_prefill"], m.get("prefix_caching", True))
judge(eng, items)
summary = {}
groups = defaultdict(list)
for it in items:
    groups[(it["variant"], it["model"])].append(it)
for key in sorted(groups):
    xs = groups[key]; ok = [x for x in xs if x["atomicity"]]; lab = Counter(x["atomicity"] for x in ok)
    g = gen[key]; mean = lambda k: sum(x[k] for x in xs if x[k]) / max(1, sum(1 for x in xs if x[k]))
    summary["/".join(key)] = {
        "problems": len(g), "correct": sum(r["correct"] for r in g),
        "answered": sum(r["status"] == "answer" for r in g),
        "steps_median": sorted(len(r["chain"]) for r in g)[len(g) // 2],
        "judged": len(xs), **{f"%{k}": round(100 * lab[k] / max(1, len(ok)), 1) for k in ("ATOMIC", "FRAGMENT", "BUNDLED")},
        **{k: round(mean(k), 2) for k in ("forward_progress", "substantiveness", "coherence")}}
    print("RESULT " + "/".join(key) + " " + json.dumps(summary["/".join(key)]), flush=True)
for it in items:
    del it["history"]
json.dump({"summary": summary, "items": items}, open(out_path, "w"))
