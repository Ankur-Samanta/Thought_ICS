"""MV@k sampling baseline under the budget-study protocol (no verification, no localization).

k independent samples per problem under the same prompt, chat template, sampling settings and
16k budget as the Thought-ICS first attempt (the step-per-paragraph prompt; a single uninterrupted
generation is distributed exactly as a thought-by-thought chain joined, since each thought call
continues the exact tokens so far). The final answer is the most common \\boxed{} answer; math
answers are pooled by math_verify equivalence, letter answers by exact match; ties go to the
earliest sample. k defaults to the 10 attempts ICS is allowed.

    python -m thought_ics.budget_study.majority_vote --model gptoss20b --dataset aime --n-problems 100
"""
import argparse
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

from thought_ics.budget_study import protocol as P
from thought_ics.budget_study.run_study import MODELS, Engine, boxed_answer, is_correct, sp
from thought_ics.datasets import load_dataset_by_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
MATH = {"amc23", "aime", "math500"}


def same(a: str, b: str, dataset: str) -> bool:
    if a == b:
        return True
    if dataset not in MATH or "NO ANSWER" in (a, b):
        return False
    try:
        from math_verify import parse, verify
        A, B = parse("$" + a + "$"), parse("$" + b + "$")
        return bool(A and B and verify(A, B))
    except Exception:  # noqa: BLE001
        return False


def vote(answers, dataset):
    """Most common answer (equivalence classes in first-seen order; ties -> earliest class)."""
    classes = []  # [representative, count]
    for a in answers:
        if a == "NO ANSWER":
            continue
        for c in classes:
            if same(c[0], a, dataset):
                c[1] += 1
                break
        else:
            classes.append([a, 1])
    if not classes:
        return "NO ANSWER", 0
    best = max(classes, key=lambda c: c[1])  # max keeps the first maximal (earliest) class
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--k", type=int, default=P.MAX_ICS_ITERATIONS)
    ap.add_argument("--temp", type=float, default=P.GENERATION_TEMP)
    ap.add_argument("--top-p", type=float, default=P.TOP_P)
    ap.add_argument("--top-k", type=int, default=P.TOP_K)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments_budget_study")
    ap.add_argument("--method", default="mv", help=argparse.SUPPRESS)  # accepted for submit.sh compatibility
    cfg = ap.parse_args()
    m = MODELS[cfg.model]
    problems = load_dataset_by_name(cfg.dataset, n_problems=cfg.n_problems, level=cfg.level, seed=cfg.seed)
    ds_tag = f"{cfg.dataset}_L{cfg.level}" if cfg.level else cfg.dataset
    run_dir = Path(cfg.out_dir) / f"{cfg.model}_{ds_tag}_mv{cfg.k}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        commit = None
    (run_dir / "config.json").write_text(json.dumps({**vars(cfg), "hf": m["hf"], "tp": m["tp"],
                                                     "max_model_len": m["max_model_len"],
                                                     "response_length": m["response_length"],
                                                     "final_prefill": m["final_prefill"], "git_commit": commit}, indent=2))
    eng = Engine(m["hf"], m["tp"], m["max_model_len"], cfg.seed, cfg.gpu_mem, m["final_prefill"],
                 m.get("prefix_caching", True))
    t0 = time.time()
    ids = [eng.gen_prompt_ids(P.THOUGHT_TEMPLATE.format(question=p["problem"])) for p in problems]
    rep = [i for i in ids for _ in range(cfg.k)]
    params = [sp(cfg, max_tokens=max(1, min(m["response_length"], eng.max_model_len - len(i))),
                 temperature=cfg.temp) for i in rep]
    outs = eng.generate(rep, params)
    recs, n_ok = [], 0
    for q, p in enumerate(problems):
        samples = outs[q * cfg.k:(q + 1) * cfg.k]
        answers = [boxed_answer(o["text"]) for o in samples]
        voted, count = vote(answers, cfg.dataset)
        correct = is_correct(f"\\boxed{{{voted}}}", p["answer"]) if voted != "NO ANSWER" else False
        n_ok += correct
        recs.append({"problem_id": p["unique_id"], "ground_truth": p["answer"], "answers": answers,
                     "n_tokens": [len(o["ids"]) for o in samples],
                     "budget_hits": sum(o["finish_reason"] == "length" for o in samples),
                     "voted": voted, "votes": count, "correct": correct,
                     "sample_correct": [is_correct(o["text"], p["answer"]) for o in samples]})
    with open(run_dir / "mv_results.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    summary = {"n_problems": len(recs), "k": cfg.k, "mv_accuracy_strict": n_ok / len(recs),
               "budget_hit_rate": sum(r["budget_hits"] for r in recs) / (len(recs) * cfg.k),
               "wall_seconds": time.time() - t0}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"summary: {json.dumps(summary)}")
    logger.info(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
