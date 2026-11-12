"""Sample first-attempt Thought chains from the final runs for thought-quality judging.

    python scripts/budget_study/quality_build.py <full_dir> <out_dir> [per_dataset]

Writes <out_dir>/final/<model>.jsonl in the format judge_variants.py reads (question, chain,
status, correct), per_dataset chains per (model, dataset), seed 0. Judge with:
    python scripts/budget_study/judge_variants.py <out_dir> <out.json>
"""
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from thought_ics.datasets import load_dataset_by_name  # noqa: E402

full, out = Path(sys.argv[1]), Path(sys.argv[2])
k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
rnd = random.Random(0)
rows = defaultdict(list)
for d in sorted(glob.glob(str(full / "*_shared_fbattempt_*/"))):
    d = Path(d)
    if not (d / "thought_results.jsonl").exists():
        continue
    c = json.load(open(d / "config.json"))
    probs = {p["unique_id"]: p["problem"] for p in
             load_dataset_by_name(c["dataset"], n_problems=c["n_problems"], level=c.get("level"), seed=42)}
    recs = [json.loads(l) for l in open(d / "thought_results.jsonl")]
    for r in rnd.sample(recs, min(k, len(recs))):
        it0 = r["iterations"][0]
        rows[c["model"]].append({"model": c["model"], "variant": "final", "dataset": c["dataset"],
                                 "problem_id": r["problem_id"], "question": probs[r["problem_id"]],
                                 "ground_truth": r["ground_truth"], "chain": it0["chain"],
                                 "status": it0["gen_status"], "n_tokens": it0["n_tokens"],
                                 "correct": it0["correct"]})
(out / "final").mkdir(parents=True, exist_ok=True)
for m, xs in rows.items():
    with open(out / "final" / f"{m}.jsonl", "w") as f:
        for x in xs:
            f.write(json.dumps(x) + "\n")
    print(m, len(xs), "chains")
