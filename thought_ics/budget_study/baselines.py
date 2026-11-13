"""Self-Refine and CoVe under the budget-study protocol.

The method logic and prompts are the paper's implementations (thought_ics/baselines/third_party,
identical to TREE/3p_baselines), imported verbatim: Self-Refine (Madaan et al. 2023; generic
feedback per Huang et al. 2024) and factored CoVe (Dhuliawala et al. 2023). What changes is only
what the study fixes for every method:
  - same engine, chat template, gpt-oss harmony handling and sampling (temp 0.5, top-p 0.9, top-k 50);
  - same first attempt: the joined Thought-ICS initial chain from the matching --self-verify run,
    so lift is paired with Thought-ICS-A / Token-ICS on every problem;
  - uniform non-binding budgets: every call gets 16k tokens (clamped to the 32k context);
  - Self-Refine runs up to 10 iterations (= Thought-ICS's 10 attempts). Every iteration is logged,
    so any shorter cap (e.g. the original 4) is read off the same trajectory exactly.
  - CoVe: one pass, as designed.
Self-Refine's YES/NO is read by the paper's parser after stripping markdown around the leading
verdict (see feedback_verdict). Calls that evaluate (Self-Refine feedback, CoVe planning and verification answers) are issued like
the study's verifier (plain chat turn; gpt-oss may reason in its analysis channel, and only the
final-channel message is used downstream). Calls that write a solution (refine, CoVe final) are
issued like the study's generations (final-channel prefill for gpt-oss).

    python -m thought_ics.budget_study.baselines --model gptoss20b --dataset aime --n-problems 100
"""
import argparse
import json
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from thought_ics.baselines.third_party import chain_of_verification as CV
from thought_ics.baselines.third_party import self_refine as SR
from thought_ics.budget_study import protocol as P
from thought_ics.budget_study.run_study import MODELS, Engine, boxed_answer, is_correct, sp
from thought_ics.datasets import load_dataset_by_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FINAL_MARK = "<|channel|>final<|message|>"


def final_message(text: str) -> str:
    """gpt-oss: the final-channel message (analysis reasoning dropped). Others: unchanged."""
    if FINAL_MARK in text:
        text = text.rsplit(FINAL_MARK, 1)[1]
        for end in ("<|return|>", "<|end|>"):
            text = text.split(end, 1)[0]
    return text


def feedback_verdict(text: str):
    """The paper's Self-Refine parser (SR.parse_feedback, verbatim), applied after removing the
    formatting around the leading YES/NO that the prompt asks for: markdown emphasis/headers
    ("**NO**", "### YES") and a leading "Answer:" / "1." label. Without this, "**Answer: NO** ...
    fundamental mistakes" never hits the startswith check and falls through to the phrase list
    (which matched "correct solution" -> YES)."""
    t = re.sub(r"[*_#`>]+", "", text).strip()
    t = re.sub(r"^(?:\d+[.):]\s*|answer\s*[:\-]\s*|feedback\s*[:\-]\s*)+", "", t, flags=re.I).strip()
    return SR.parse_feedback(t)


def call(eng, cfg, texts, solution: bool):
    """Batched calls. solution=True: written like a generation; False: like the verifier."""
    if not texts:
        return []
    ids = [eng.gen_prompt_ids(t) if solution else eng.chat_ids(t) for t in texts]
    params = [sp(cfg, max_tokens=max(1, min(cfg.budget, eng.max_model_len - len(i))),
                 temperature=cfg.temp, skip_special_tokens=solution) for i in ids]
    outs = eng.generate(ids, params)
    res = []
    for o, i in zip(outs, ids):
        msg = o["text"] if solution else final_message(o["text"])
        res.append({"raw": o["text"], "text": msg.strip(), "finish_reason": o["finish_reason"],
                    "n_tokens": len(o["ids"]), "prompt_tokens": len(i)})
    return res


def load_init(path: Path, problems):
    by_id = {}
    for line in open(path / "thought_results.jsonl"):
        r = json.loads(line)
        by_id[r["problem_id"]] = r
    return [("".join(by_id[p["unique_id"]]["iterations"][0]["chain"])).strip() for p in problems]


def sol_record(text, gt, **kw):
    return {**kw, "solution": text, "answer": boxed_answer(text), "correct": is_correct(text, gt)}


def run_self_refine(eng, cfg, problems, init):
    recs = [{"problem_id": p["unique_id"], "ground_truth": p["answer"],
             "iterations": [sol_record(init[k], p["answer"], iteration=0)], "stopped_early": False}
            for k, p in enumerate(problems)]
    cur = list(init)
    active = list(range(len(problems)))
    for it in range(cfg.max_iterations):
        fbs = call(eng, cfg, [SR.FEEDBACK_PROMPT.format(problem=problems[i]["problem"], solution=cur[i])
                              for i in active], solution=False)
        refine = []
        for i, fb in zip(active, fbs):
            ok, why = feedback_verdict(fb["text"])
            recs[i]["iterations"][-1]["feedback"] = {**fb, "says_correct": ok, "parse": why}
            if ok:
                recs[i]["stopped_early"] = True
            else:
                refine.append((i, fb["text"]))
        outs = call(eng, cfg, [SR.REFINE_PROMPT.format(problem=problems[i]["problem"], solution=cur[i],
                                                       feedback=f) for i, f in refine], solution=True)
        for (i, _), o in zip(refine, outs):
            cur[i] = o["text"]
            recs[i]["iterations"].append(sol_record(o["text"], recs[i]["ground_truth"], iteration=it + 1,
                                                    finish_reason=o["finish_reason"], n_tokens=o["n_tokens"]))
        active = [i for i, _ in refine]
        logger.info(f"[self_refine] iter {it + 1}: {len(active)} refined, "
                    f"accuracy {sum(r['iterations'][-1]['correct'] for r in recs)}/{len(recs)}")
        if not active:
            break
    for r in recs:
        r["final_correct"] = r["iterations"][-1]["correct"]
    return recs


def run_cove(eng, cfg, problems, init):
    n = len(problems)
    plans = call(eng, cfg, [CV.PLAN_VERIFICATION_PROMPT.format(problem=p["problem"], baseline=init[k])
                            for k, p in enumerate(problems)], solution=False)
    qs = [CV.parse_verification_questions(pl["text"])[:4] for pl in plans]
    flat = [(k, q) for k in range(n) for q in qs[k]]
    ans = call(eng, cfg, [CV.EXECUTE_VERIFICATION_PROMPT.format(question=q) for _, q in flat], solution=False)
    by = [[] for _ in range(n)]
    for (k, q), a in zip(flat, ans):
        by[k].append((q, a))
    finals = []
    for k, p in enumerate(problems):
        vt = "".join(f"Q{i}: {q}\nA{i}: {a['text']}\n\n" for i, (q, a) in enumerate(by[k], 1))
        if not vt.strip():
            vt = "(No verification questions generated)"
        finals.append(CV.FINAL_RESPONSE_PROMPT.format(problem=p["problem"], baseline=init[k], verifications=vt))
    outs = call(eng, cfg, finals, solution=True)
    recs = []
    for k, p in enumerate(problems):
        recs.append({"problem_id": p["unique_id"], "ground_truth": p["answer"],
                     "iterations": [sol_record(init[k], p["answer"], iteration=0),
                                    sol_record(outs[k]["text"], p["answer"], iteration=1,
                                               finish_reason=outs[k]["finish_reason"], n_tokens=outs[k]["n_tokens"])],
                     "plan": plans[k], "verifications": [{"question": q, **a} for q, a in by[k]]})
        recs[-1]["final_correct"] = recs[-1]["iterations"][-1]["correct"]
    logger.info(f"[cove] accuracy {sum(r['final_correct'] for r in recs)}/{n}")
    return recs


def summarize(method, recs):
    n = len(recs)
    calls = [it["feedback"] for r in recs for it in r["iterations"] if "feedback" in it] if method == "self_refine" \
        else [r["plan"] for r in recs] + [v for r in recs for v in r["verifications"]]
    sols = [it for r in recs for it in r["iterations"][1:]]
    out = {"method": method, "n_problems": n,
           "init_accuracy": sum(r["iterations"][0]["correct"] for r in recs) / n,
           "final_accuracy": sum(r["final_correct"] for r in recs) / n,
           "eval_calls": len(calls), "eval_budget_hits": sum(c["finish_reason"] == "length" for c in calls),
           "solution_calls": len(sols), "solution_budget_hits": sum(s["finish_reason"] == "length" for s in sols)}
    if method == "self_refine":
        out["stopped_early"] = sum(r["stopped_early"] for r in recs)
        out["feedback_parse"] = {}
        for c in calls:
            out["feedback_parse"][c["parse"]] = out["feedback_parse"].get(c["parse"], 0) + 1
    else:
        out["mean_questions"] = sum(len(r["verifications"]) for r in recs) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--method", choices=["self_refine", "cove", "both"], default="both")
    ap.add_argument("--init-from", default=None,
                    help="--self-verify run dir with thought_results.jsonl (default: latest matching run in --out-dir)")
    ap.add_argument("--max-iterations", type=int, default=P.MAX_ICS_ITERATIONS, help="Self-Refine (= ICS attempts)")
    ap.add_argument("--budget", type=int, default=None, help="per call; default: the model's generation budget")
    ap.add_argument("--temp", type=float, default=P.GENERATION_TEMP)
    ap.add_argument("--top-p", type=float, default=P.TOP_P)
    ap.add_argument("--top-k", type=int, default=P.TOP_K)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments_budget_study")
    cfg = ap.parse_args()
    m = MODELS[cfg.model]
    cfg.budget = cfg.budget or m["response_length"]
    ds_tag = f"{cfg.dataset}_L{cfg.level}" if cfg.level else cfg.dataset
    if cfg.init_from is None:
        cands = sorted(d for d in Path(cfg.out_dir).glob(f"{cfg.model}_{ds_tag}_selfverify_shared_fbattempt_*")
                       if (d / "thought_results.jsonl").exists())
        if not cands:
            raise SystemExit(f"no finished self-verify run for {cfg.model} {ds_tag} in {cfg.out_dir}")
        cfg.init_from = str(cands[-1])
    problems = load_dataset_by_name(cfg.dataset, n_problems=cfg.n_problems, level=cfg.level, seed=cfg.seed)
    init = load_init(Path(cfg.init_from), problems)
    run_dir = Path(cfg.out_dir) / f"{cfg.model}_{ds_tag}_baselines_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        commit = None
    config = {**vars(cfg), "hf": m["hf"], "tp": m["tp"], "max_model_len": m["max_model_len"],
              "final_prefill": m["final_prefill"], "git_commit": commit}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info(f"Config: {json.dumps(config)}")
    eng = Engine(m["hf"], m["tp"], m["max_model_len"], cfg.seed, cfg.gpu_mem, m["final_prefill"],
                 m.get("prefix_caching", True))
    summaries = {}
    for method in (["self_refine", "cove"] if cfg.method == "both" else [cfg.method]):
        t0 = time.time()
        recs = (run_self_refine if method == "self_refine" else run_cove)(eng, cfg, problems, init)
        with open(run_dir / f"{method}_results.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        summaries[method] = {**summarize(method, recs), "wall_seconds": time.time() - t0}
        logger.info(f"[{method}] summary: {json.dumps(summaries[method])}")
        (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    logger.info(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
