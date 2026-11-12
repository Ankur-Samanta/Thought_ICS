#!/usr/bin/env python3
"""Budget-controlled Thought-ICS vs Token-ICS study (oracle-verified ICS, L1/L2).

Thought-ICS follows the SRPO protocol (see ``protocol.py``). Token-ICS mirrors every
knob: same total generation budget, same localization budget, temperature and
chat-template handling, same parse fallback order, same empty-output handling. Both
methods run in one job on one vLLM engine, batched across problems.

Usage:
    python -m thought_ics.budget_study.run_study --model qwen32b --dataset aime \
        --n-problems 100 --method both --out-dir experiments_budget_study
"""

import argparse
import json
import logging
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from thought_ics.budget_study import protocol as P
from thought_ics.datasets import load_dataset_by_name, normalize_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("budget_study")

# final_prefill: gpt-oss (harmony format) starts its reply directly in the final channel, so
# all solution reasoning is visible and in the format under study for both methods; the
# localizer keeps the model's normal analysis channel.
# prefix_caching: off for gpt-oss. With vLLM 0.11.2 on A100, gpt-oss with TP>1 AND prefix
# caching degenerates to token 0 ("!!!!") on continuation calls (25/30 at TP=2); TP=2 without
# caching and TP=1 either way are clean (0/30) -- scripts/budget_study/diag_gptoss_continuation.py.
# Budgets are uniform across models AND methods (scaling analysis): the largest the
# smallest context allows. qwen32b has a 32k context, so every model gets a 32k window,
# 16k generation (whole solution / chain, incl. resamples) and an 8k localizer (a 16k chain
# in the localizer prompt + 8k output fits). Every run logs budget hits to verify.
UNIFORM = {"max_model_len": 32768, "response_length": 16384, "localization_max_tokens": 8192}
MODELS = {
    "qwen32b": {"hf": "Qwen/Qwen2.5-32B-Instruct", "tp": 4, **UNIFORM,
                "final_prefill": False},
    "gptoss20b": {"hf": "openai/gpt-oss-20b", "tp": 1, **UNIFORM,
                  "final_prefill": True, "prefix_caching": False},
    "gptoss120b": {"hf": "openai/gpt-oss-120b", "tp": 4, **UNIFORM,
                   "final_prefill": True, "prefix_caching": False},
}
HARMONY_FINAL_HEADER = "<|channel|>final<|message|>"


def boxed_answer(text: str) -> str:
    """Last \\boxed{} content (paper grading uses normalize_answer equality)."""
    b = P._last_boxed(text or "")
    return b if b is not None else "NO ANSWER"


def is_correct(text: str, gt: str) -> bool:
    return normalize_answer(boxed_answer(text)) == normalize_answer(gt)


# --- Engine -------------------------------------------------------------------------
class Engine:
    def __init__(self, hf: str, tp: int, max_model_len: int, seed: int, gpu_mem: float,
                 final_prefill: bool = False, prefix_caching: bool = True):
        from vllm import LLM
        self.llm = LLM(model=hf, tensor_parallel_size=tp, max_model_len=max_model_len,
                       seed=seed, gpu_memory_utilization=gpu_mem,
                       enable_prefix_caching=prefix_caching, trust_remote_code=True)
        self.tok = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        self.final_prefill = final_prefill

    def generate(self, prompt_ids: List[List[int]], params: list) -> List[dict]:
        from vllm.inputs import TokensPrompt
        if not prompt_ids:
            return []
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids],
                                 params, use_tqdm=False)
        return [{"ids": list(o.outputs[0].token_ids), "text": o.outputs[0].text,
                 "finish_reason": o.outputs[0].finish_reason,
                 "stop_reason": o.outputs[0].stop_reason} for o in outs]

    def gen_prompt_ids(self, text: str) -> List[int]:
        """Chat-templated user turn, open assistant reply (final channel for gpt-oss).

        Thoughts / the CoT solution are generated inside this one assistant reply, and
        resamples continue it from the kept prefix.
        """
        ids = self.chat_ids(text)
        if self.final_prefill:
            ids += self.tok.encode(HARMONY_FINAL_HEADER, add_special_tokens=False)
        return ids

    def chat_ids(self, text: str) -> List[int]:
        ids = self.tok.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return list(ids)


def sp(cfg, *, max_tokens: int, temperature: float, stop: Optional[List[str]] = None,
       skip_special_tokens: bool = True):
    """Generation text drops special tokens (EOS etc.); token ids are kept verbatim.
    Localization keeps them (skip_special_tokens=False) so channel structure is logged."""
    from vllm import SamplingParams
    return SamplingParams(temperature=temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                          max_tokens=max_tokens, stop=stop, include_stop_str_in_output=bool(stop),
                          skip_special_tokens=skip_special_tokens)


# --- Generation (batched) ----------------------------------------------------------
def gen_thought_chains(eng: Engine, cfg, jobs: List[dict]) -> List[dict]:
    """SRPO ``_generate_thoughts_core`` rules, batched. job: prompt_ids [+ prefix]."""
    states = [{"prompt_ids": j["prompt_ids"], "ids": list(j.get("prefix_ids", [])),
               "thoughts": list(j.get("prefix_thoughts", [])),
               "bounds": list(j.get("prefix_bounds", [])),
               "prefix_n": len(j.get("prefix_thoughts", [])),
               "status": None, "thought_cap_hits": 0} for j in jobs]
    while True:
        active = []
        for s in states:
            if s["status"] is not None:
                continue
            if len(s["thoughts"]) >= cfg.max_thoughts:
                s["status"] = "max_thoughts"
            elif len(s["ids"]) + cfg.max_tokens_per_thought > cfg.response_length:
                s["status"] = "budget"
            elif (len(s["prompt_ids"]) + len(s["ids"]) + cfg.max_tokens_per_thought
                  > eng.max_model_len):
                s["status"] = "context"
            else:
                active.append(s)
        if not active:
            break
        params = sp(cfg, max_tokens=cfg.max_tokens_per_thought, temperature=cfg.generation_temp,
                    stop=P.THOUGHT_STOPS)
        outs = eng.generate([s["prompt_ids"] + s["ids"] for s in active], [params] * len(active))
        for s, o in zip(active, outs):
            if not o["ids"]:
                s["status"] = "empty_output"
                continue
            start = len(s["ids"])
            s["ids"].extend(o["ids"])
            if not P.has_content(o["text"]) and s["thoughts"]:
                # separator / whitespace-only unit: part of the previous thought, not a step
                s["thoughts"][-1] += o["text"]
                s["bounds"][-1] = (s["bounds"][-1][0], len(s["ids"]))
            else:
                s["bounds"].append((start, len(s["ids"])))
                s["thoughts"].append(o["text"])
                s["thought_cap_hits"] += o["finish_reason"] == "length"
            # Answer check on the whole chain: a \\boxed{...} can span a thought boundary.
            if P._last_boxed("".join(s["thoughts"])) is not None:
                s["status"] = "answer"
            elif o["finish_reason"] == "stop" and o["stop_reason"] is None:
                # End of the assistant reply without an answer. In chat format the reply
                # is over, so the chain ends here (same as Token-ICS stopping at EOS).
                s["status"] = "eos_no_answer"
    for s in states:
        s["text"] = "".join(s["thoughts"])
        s["n_tokens"] = len(s["ids"])
    return states


def gen_token_solutions(eng: Engine, cfg, jobs: List[dict]) -> List[dict]:
    """One CoT call per job with the same total budget (response_length)."""
    states, todo = [], []
    for j in jobs:
        prefix_ids = list(j.get("prefix_ids", []))
        budget = min(cfg.response_length - len(prefix_ids),
                     eng.max_model_len - len(j["prompt_ids"]) - len(prefix_ids))
        s = {"prompt_ids": j["prompt_ids"], "prefix_ids": prefix_ids,
             "prefix_text": j.get("prefix_text", ""), "budget": budget, "status": None}
        states.append(s)
        if budget > 0:
            todo.append(s)
        else:
            s.update(status="budget", gen_ids=[], gen_text="", finish_reason=None)
    params = [sp(cfg, max_tokens=s["budget"], temperature=cfg.generation_temp) for s in todo]
    for s, o in zip(todo, eng.generate([s["prompt_ids"] + s["prefix_ids"] for s in todo], params)):
        s.update(gen_ids=o["ids"], gen_text=o["text"], finish_reason=o["finish_reason"])
    for s in states:
        s["ids"] = s["prefix_ids"] + s["gen_ids"]
        s["text"] = s["prefix_text"] + s["gen_text"]
        s["n_tokens"] = len(s["ids"])
        if s["status"] is None:
            if "\\boxed{" in s["text"]:
                s["status"] = "answer"
            elif s["finish_reason"] == "length":
                s["status"] = "budget"
            elif not s["gen_text"].strip():
                s["status"] = "empty_output"
            else:
                s["status"] = "eos_no_answer"
    return states


def localize(eng: Engine, cfg, prompt_texts: List[str]) -> List[dict]:
    """Chat-templated localization call, no stop tokens (SRPO ``_localize_error``)."""
    ids = [eng.chat_ids(t) for t in prompt_texts]
    params = [sp(cfg, max_tokens=max(1, min(cfg.localization_max_tokens, eng.max_model_len - len(i))),
                 temperature=cfg.localization_temp, skip_special_tokens=False) for i in ids]
    outs = eng.generate(ids, params)
    return [{"response": o["text"], "finish_reason": o["finish_reason"],
             "n_tokens": len(o["ids"]), "prompt_tokens": len(i)} for o, i in zip(outs, ids)]


# --- ICS loop -----------------------------------------------------------------------
def is_empty(method: str, g: dict) -> bool:
    return len(g["thoughts"]) == 0 if method == "thought" else not g["text"].strip()


def gen_record(method: str, g: dict, gt: str) -> dict:
    rec = {"n_tokens": g["n_tokens"], "gen_status": g["status"],
           "answer": boxed_answer(g["text"]), "correct": is_correct(g["text"], gt)}
    if method == "thought":
        rec.update(chain=g["thoughts"], n_thoughts=len(g["thoughts"]),
                   n_new_thoughts=len(g["thoughts"]) - g["prefix_n"],
                   thought_cap_hits=g["thought_cap_hits"])
    else:
        rec.update(solution=g["text"], finish_reason=g["finish_reason"])
    return rec


def run_method(eng: Engine, cfg, method: str, problems: List[dict]) -> List[dict]:
    gen = gen_thought_chains if method == "thought" else gen_token_solutions
    template = P.THOUGHT_TEMPLATE if method == "thought" else P.TOKEN_TEMPLATE
    prompt_ids = [eng.gen_prompt_ids(template.format(question=p["problem"])) for p in problems]
    records = [{"problem_id": p.get("unique_id", str(k)), "ground_truth": p["answer"],
                "empty_retries": 0, "iterations": []} for k, p in enumerate(problems)]

    # Initial generation; SRPO regenerates zero-thought chains (bounded retries, both methods).
    current: Dict[int, dict] = {}
    pending = list(range(len(problems)))
    for attempt in range(cfg.max_empty_retries + 1):
        outs = gen(eng, cfg, [{"prompt_ids": prompt_ids[i]} for i in pending])
        still = []
        for i, g in zip(pending, outs):
            if is_empty(method, g) and attempt < cfg.max_empty_retries:
                records[i]["empty_retries"] += 1
                still.append(i)
            else:
                current[i] = g
        pending = still
        if not pending:
            break

    done, stopped = {}, {}
    for i, r in enumerate(records):
        rec = gen_record(method, current[i], r["ground_truth"])
        rec["iteration"] = 0
        r["iterations"].append(rec)
        done[i] = rec["correct"]
    logger.info(f"[{method}] init accuracy {sum(done.values())}/{len(records)}")

    for it in range(1, cfg.max_ics_iterations + 1):
        active = [i for i in range(len(records)) if not done[i] and not stopped.get(i)]
        if not active:
            break
        if method == "thought":
            texts = [P.thought_localization_prompt(problems[i]["problem"], current[i]["thoughts"],
                                                   records[i]["ground_truth"], cfg.autonomy_level)
                     for i in active]
        else:
            texts = [P.token_localization_prompt(problems[i]["problem"], current[i]["text"],
                                                 records[i]["ground_truth"], cfg.autonomy_level)
                     for i in active]
        locs = localize(eng, cfg, texts)

        jobs, job_idx, loc_info = [], [], {}
        for i, text, loc in zip(active, texts, locs):
            g = current[i]
            info = {"loc_prompt": text, **loc}
            if method == "thought":
                n = len(g["thoughts"])
                if loc["n_tokens"] == 0:
                    step, status = max(1, n // 2), "empty"  # SRPO empty-output rule
                else:
                    step, status = P.parse_error_step(loc["response"], n)
                info.update(parse_status=status, error_step=step)
                if step == 0:
                    stopped[i] = True
                    loc_info[i] = info
                    continue
                keep = step - 1
                cut = g["bounds"][keep - 1][1] if keep > 0 else 0
                job = {"prompt_ids": prompt_ids[i], "prefix_ids": g["ids"][:cut],
                       "prefix_thoughts": g["thoughts"][:keep], "prefix_bounds": g["bounds"][:keep]}
                info["prefix_len"] = keep
            else:
                if loc["n_tokens"] == 0:
                    quote, status = None, "empty"
                else:
                    quote, status = P.parse_error_quote(loc["response"])
                info.update(parse_status=status, error_quote=quote)
                if quote == "":
                    stopped[i] = True
                    loc_info[i] = info
                    continue
                idx = P.find_quote(g["text"], quote) if quote else None
                if status == "empty":  # SRPO empty-output rule: middle of the trace
                    idx = len(g["text"]) // 2
                info["quote_found"] = idx is not None
                prefix_text = g["text"][:idx].rstrip() if idx else ""
                job = {"prompt_ids": prompt_ids[i], "prefix_text": prefix_text,
                       "prefix_ids": eng.tok.encode(prefix_text, add_special_tokens=False)
                       if prefix_text else []}
                info.update(truncation_idx=idx, prefix_len=len(job["prefix_ids"]))
            jobs.append(job)
            job_idx.append(i)
            loc_info[i] = info

        for i in active:
            if stopped.get(i):
                records[i]["iterations"].append({"iteration": it, "localization": loc_info[i],
                                                 "stopped": "no_error_found"})
        for i, g in zip(job_idx, gen(eng, cfg, jobs)):
            rec = gen_record(method, g, records[i]["ground_truth"])
            rec.update(iteration=it, localization=loc_info[i])
            records[i]["iterations"].append(rec)
            current[i] = g
            done[i] = rec["correct"]
        logger.info(f"[{method}] iter {it}: accuracy {sum(done.values())}/{len(records)}")

    for i, r in enumerate(records):
        r["final_correct"] = done[i]
    return records


# --- Summary ------------------------------------------------------------------------
def summarize(method: str, records: List[dict]) -> dict:
    n = len(records)
    gens = [it for r in records for it in r["iterations"] if "gen_status" in it]
    locs = [it["localization"] for r in records for it in r["iterations"] if "localization" in it]
    init = sum(r["iterations"][0]["correct"] for r in records)
    final = sum(r["final_correct"] for r in records)
    budget_status = ("budget", "max_thoughts", "context")
    s = {
        "method": method, "n_problems": n,
        "init_accuracy": init / n, "final_accuracy": final / n, "lift": (final - init) / n,
        "initial_gen_status": dict(Counter(r["iterations"][0]["gen_status"] for r in records)),
        "initial_no_answer_from_budget": sum(r["iterations"][0]["gen_status"] in budget_status
                                             for r in records) / n,
        "all_gen_status": dict(Counter(g["gen_status"] for g in gens)),
        "empty_retries": sum(r["empty_retries"] for r in records),
        "n_localizations": len(locs),
        "loc_finish_length_rate": (sum(l["finish_reason"] == "length" for l in locs) / len(locs))
        if locs else None,
        "loc_parse_status": dict(Counter(l["parse_status"] for l in locs)),
    }
    if method == "thought":
        s["thought_cap_hit_rate"] = (sum(g["thought_cap_hits"] for g in gens)
                                     / max(1, sum(g["n_new_thoughts"] for g in gens)))
    else:
        tl = [l for l in locs if l.get("error_quote")]
        s["quote_not_found_rate"] = (sum(not l["quote_found"] for l in tl) / len(tl)) if tl else None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--method", choices=["thought", "token", "both"], default="both")
    ap.add_argument("--autonomy-level", type=int, choices=[1, 2], default=P.AUTONOMY_LEVEL)
    ap.add_argument("--max-thoughts", type=int, default=P.MAX_THOUGHTS)
    ap.add_argument("--max-tokens-per-thought", type=int, default=P.MAX_TOKENS_PER_THOUGHT)
    ap.add_argument("--response-length", type=int, default=None,
                    help="default: per-model value in MODELS")
    ap.add_argument("--localization-max-tokens", type=int, default=None,
                    help="default: per-model value in MODELS")
    ap.add_argument("--localization-temp", type=float, default=P.LOCALIZATION_TEMP)
    ap.add_argument("--generation-temp", type=float, default=P.GENERATION_TEMP)
    ap.add_argument("--top-p", type=float, default=P.TOP_P)
    ap.add_argument("--top-k", type=int, default=P.TOP_K)
    ap.add_argument("--max-ics-iterations", type=int, default=P.MAX_ICS_ITERATIONS)
    ap.add_argument("--max-empty-retries", type=int, default=P.MAX_EMPTY_RETRIES)
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments_budget_study")
    cfg = ap.parse_args()

    m = MODELS[cfg.model]
    if cfg.response_length is None:
        cfg.response_length = m["response_length"]
    if cfg.localization_max_tokens is None:
        cfg.localization_max_tokens = m["localization_max_tokens"]
    tp = cfg.tp or m["tp"]
    max_model_len = cfg.max_model_len or m["max_model_len"]
    problems = load_dataset_by_name(cfg.dataset, n_problems=cfg.n_problems, level=cfg.level, seed=cfg.seed)
    ds_tag = f"{cfg.dataset}_L{cfg.level}" if cfg.level else cfg.dataset
    run_dir = Path(cfg.out_dir) / f"{cfg.model}_{ds_tag}_L{cfg.autonomy_level}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                         text=True).strip()
    except Exception:
        commit = None
    config = {**vars(cfg), "hf": m["hf"], "tp": tp, "max_model_len": max_model_len,
              "final_prefill": m["final_prefill"], "chat_template": True,
              "prefix_caching": m.get("prefix_caching", True),
              "git_commit": commit, "protocol": "SRPO thought-ICS (c99cdbb) + matched Token-ICS"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info(f"Config: {json.dumps(config)}")

    eng = Engine(m["hf"], tp, max_model_len, cfg.seed, cfg.gpu_mem, m["final_prefill"],
                 m.get("prefix_caching", True))
    methods = ["thought", "token"] if cfg.method == "both" else [cfg.method]
    summaries = {}
    for method in methods:
        t0 = time.time()
        records = run_method(eng, cfg, method, problems)
        with open(run_dir / f"{method}_results.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        summaries[method] = {**summarize(method, records), "wall_seconds": time.time() - t0}
        logger.info(f"[{method}] summary: {json.dumps(summaries[method])}")
        (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    logger.info(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
