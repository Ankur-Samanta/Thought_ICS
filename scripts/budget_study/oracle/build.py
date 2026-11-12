"""Build oracle-labeling batches from the final (shared-init) runs.

Both methods localize in the SAME first attempt (Token-ICS text = joined Thought-ICS chain),
so one oracle label per wrong first attempt ("first erroneous step k") serves both methods.

    python scripts/budget_study/oracle/build.py <full_dir> <out_dir> [per_model] [n_batches]

Deterministic (seed 0): rerunning on the same results gives the same batches. Writes
batch_<k>.json and manifest.json (sampling settings, source runs, item ids per batch).
"""
import glob
import json
import random
import re
import sys
from pathlib import Path

full, out = Path(sys.argv[1]), Path(sys.argv[2])
per_model = int(sys.argv[3]) if len(sys.argv) > 3 else 30
n_batches = int(sys.argv[4]) if len(sys.argv) > 4 else 3
QUOTA = {"amc23": 6, "aime": 8, "math500": 8, "mathqa": 8}  # math datasets: errors are checkable
random.seed(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from thought_ics.datasets import load_dataset_by_name  # noqa: E402


def equivalent(answer, gt, dataset):
    """True if the answer is actually right: math_verify equivalence for math datasets
    (catches 5.5 vs \\frac{11}{2}, x=5 vs 5, 120^\\circ vs 120), formatting-insensitive match
    otherwise. Used to skip grader false negatives."""
    if dataset in ("amc23", "aime", "math500"):
        try:
            from math_verify import parse, verify
            A, G = parse("$" + answer + "$"), parse("$" + gt + "$")
            if A and G and verify(G, A):
                return True
        except Exception:  # noqa: BLE001
            pass
    return loose(answer) == loose(gt)


def loose(a):  # formatting-insensitive answer form
    a = a or ""
    a = re.sub(r"\^\s*\{?\\circ\}?|°|\\%|%|\\\$|\$|\\,|\\!|\\;|\s", "", a)
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\(left|right)", "", a)
    return a.lower().strip(".")


items, sources = [], {}
for model in ("qwen32b", "gptoss20b", "gptoss120b"):
    pool = []
    for ds, q in QUOTA.items():
        runs = glob.glob(str(full / f"{model}_{ds}*_shared_fbattempt_*/thought_results.jsonl"))
        if not runs:
            print(f"missing {model} {ds}")
            continue
        cfg = json.load(open(Path(runs[0]).parent / "config.json"))
        sources[f"{model}/{ds}"] = str(Path(runs[0]).parent)
        probs = {p["unique_id"]: p["problem"] for p in
                 load_dataset_by_name(cfg["dataset"], n_problems=cfg["n_problems"], level=cfg.get("level"), seed=42)}
        wrong = []
        for line in open(runs[0]):
            r = json.loads(line)
            it0 = r["iterations"][0]
            if it0["correct"] or it0["answer"] == "NO ANSWER" or equivalent(it0["answer"], r["ground_truth"], ds):
                continue
            if len(r["iterations"]) < 2:
                continue
            wrong.append({"id": f"{model}|{ds}|{r['problem_id']}", "model": model, "dataset": ds,
                          "problem_id": r["problem_id"], "problem": probs[r["problem_id"]],
                          "ground_truth": r["ground_truth"], "final_answer_given": it0["answer"],
                          "steps": [t.strip() for t in it0["chain"]]})
        pool += random.sample(wrong, min(q, len(wrong)))
    items += pool
random.shuffle(items)
out.mkdir(parents=True, exist_ok=True)
for b in range(n_batches):
    batch = items[b::n_batches]
    (out / f"batch_{b}.json").write_text(json.dumps(batch, indent=1))
    print(f"batch_{b}: {len(batch)} items, {sum(len(x['steps']) for x in batch)} steps")
(out / "manifest.json").write_text(json.dumps({
    "per_model_quota": QUOTA, "seed": 0, "n_batches": n_batches, "sources": sources,
    "batches": {f"batch_{b}": [x["id"] for x in items[b::n_batches]] for b in range(n_batches)}}, indent=1))
print("total", len(items), {m: sum(x['model'] == m for x in items) for m in ("qwen32b", "gptoss20b", "gptoss120b")})
