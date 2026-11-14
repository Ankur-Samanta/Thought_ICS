#!/usr/bin/env python3
"""Thought-ICS and Token-ICS: generation, verification, localization and backtrack-and-resample.

Thought-ICS generates the first attempt thought by thought; Token-ICS starts from the same first
attempt joined into one unstructured response. Both methods then iterate verification ->
localization -> resampling from the localized prefix, with the same budgets, sampling settings and
chat templates (see ``protocol.py``). Both run in one job on one vLLM engine, batched over problems.

Verification is either an oracle (the model is told when its answer is wrong; ``--verify oracle``)
or the model itself (``--verify self``). Under self-verification, ``--feedback all`` carries the
verification reasoning into localization and both reasonings into resampling (Thought-ICS-F);
``--feedback loc`` carries the verification reasoning into localization only.

    python -m thought_ics.ics --model qwen32b --dataset aime                      # oracle-verified
    python -m thought_ics.ics --model qwen32b --dataset aime --verify self       # self-verified
    python -m thought_ics.ics --model qwen32b --dataset aime --verify self --feedback all \
        --method thought --init-from <self-verified run dir>                     # Thought-ICS-F
"""

import argparse
import json
import logging
import re
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from thought_ics import protocol as P
from thought_ics.benchmarks import BENCHMARKS, load_benchmark, normalize_answer, run_tag

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("thought_ics")

# hf: Hugging Face model id; tp: tensor-parallel GPUs (sized for 40GB A100s).
# final_prefill (gpt-oss): the response is generated in the final channel after an empty analysis
# block, so all solution reasoning is visible. prefix_caching is disabled for gpt-oss: with vLLM
# 0.11.2 and tensor parallelism, prefix caching corrupts gpt-oss continuations.
MODELS = {
    "llama3b": {"hf": "meta-llama/Llama-3.2-3B-Instruct", "tp": 1, "final_prefill": False},
    "llama8b": {"hf": "meta-llama/Llama-3.1-8B-Instruct", "tp": 1, "final_prefill": False},
    "llama70b": {"hf": "meta-llama/Llama-3.3-70B-Instruct", "tp": 8, "final_prefill": False},
    "qwen7b": {"hf": "Qwen/Qwen2.5-7B-Instruct", "tp": 1, "final_prefill": False},
    "qwen14b": {"hf": "Qwen/Qwen2.5-14B-Instruct", "tp": 2, "final_prefill": False},
    "qwen32b": {"hf": "Qwen/Qwen2.5-32B-Instruct", "tp": 4, "final_prefill": False},
    "gptoss20b": {"hf": "openai/gpt-oss-20b", "tp": 1, "final_prefill": True, "prefix_caching": False},
    "gptoss120b": {"hf": "openai/gpt-oss-120b", "tp": 4, "final_prefill": True, "prefix_caching": False},
}
# run-directory tag per setting
MODE_TAG = {("oracle", None): "oracle", ("self", None): "self", ("self", "loc"): "self-fb-loc",
            ("self", "all"): "self-fb-all"}
# gpt-oss: an explicitly empty analysis block, then the final channel, so the turn has the
# harmony structure the model was trained on while all solution reasoning stays visible.
HARMONY_FINAL_HEADER = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"


FINAL_MARK = "<|channel|>final<|message|>"


def final_message(text: str) -> str:
    """gpt-oss: the final-channel message (analysis reasoning dropped); others unchanged.
    Special tokens are removed (verifier/localizer outputs keep them for logging)."""
    if FINAL_MARK in text:
        text = text.rsplit(FINAL_MARK, 1)[1]
    for t in ("<|return|>", "<|end|>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>"):
        text = text.split(t, 1)[0]
    return text


def response_text(text: str) -> str:
    """The whole response as plain text: for gpt-oss the analysis-channel reasoning followed by
    the final message (channel markup removed); for other models the response itself. Used to
    carry verifier / localizer feedback forward, whose content gpt-oss writes in its analysis
    channel (its final message is often only the boxed verdict)."""
    if "<|channel|>" not in text:
        return final_message(text)
    parts = re.findall(r"<\|channel\|>(?:analysis|commentary|final)<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|start\|>|$)",
                       text, flags=re.S)
    return "\n\n".join(p.strip() for p in parts if p.strip())


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
    """Thought-by-thought generation, batched over jobs (job: prompt_ids [+ kept prefix]).

    Each call generates one paragraph; the next call continues the same response from the exact
    tokens so far. A paragraph that only continues the previous thought (no content, a bare displayed
    equation, or after a thought that introduces it) is attached to it. The chain ends when the
    response ends, at the token budget, or at max_thoughts."""
    states = [{"prompt_ids": j["prompt_ids"], "ids": list(j.get("prefix_ids", [])),
               "thoughts": list(j.get("prefix_thoughts", [])),
               "bounds": list(j.get("prefix_bounds", [])),
               "prefix_n": len(j.get("prefix_thoughts", [])),
               "prefix_tokens": len(j.get("prefix_ids", [])),
               "status": None, "thought_cap_hits": 0} for j in jobs]
    while True:
        active, params = [], []
        for s in states:
            if s["status"] is not None:
                continue
            # Same total budget as Token-ICS: each thought may use the per-thought cap or whatever
            # remains of response_length, whichever is smaller.
            remaining = cfg.response_length - len(s["ids"])
            room = eng.max_model_len - len(s["prompt_ids"]) - len(s["ids"])
            if len(s["thoughts"]) >= cfg.max_thoughts:
                s["status"] = "max_thoughts"
            elif remaining <= 0:
                s["status"] = "budget"
            elif room <= 0:
                s["status"] = "context"
            else:
                active.append(s)
                params.append(sp(cfg, max_tokens=min(cfg.max_tokens_per_thought, remaining, room),
                                 temperature=cfg.generation_temp, stop=P.THOUGHT_STOPS))
        if not active:
            break
        outs = eng.generate([s["prompt_ids"] + s["ids"] for s in active], params)
        for s, o in zip(active, outs):
            # End of the assistant reply: any stop other than the step boundary (EOS may be
            # reported as stop_reason None or as an int EOS id from generation_config).
            eos = o["finish_reason"] == "stop" and o["stop_reason"] not in P.THOUGHT_STOPS
            start = len(s["ids"])
            s["ids"].extend(o["ids"])
            # Boundary rules (decided as each paragraph is generated; tokens unchanged): a unit
            # attaches to the previous thought if it has no content, is only a displayed
            # equation, or the previous thought only introduced it (ends with ':' / 'Thus').
            attach = bool(s["thoughts"]) and (not P.has_content(o["text"]) or P.is_display_math(o["text"])
                                              or P.opens_next(s["thoughts"][-1]))
            if not attach and P.has_content(o["text"]):
                s["bounds"].append((start, len(s["ids"])))
                s["thoughts"].append(o["text"])
                s["thought_cap_hits"] += o["finish_reason"] == "length"
            elif attach:
                # separator / whitespace-only unit, or a bare displayed equation completing the
                # previous sentence: part of the previous thought, not a step of its own
                s["thoughts"][-1] += o["text"]
                s["bounds"][-1] = (s["bounds"][-1][0], len(s["ids"]))
            # Like Token-ICS, the chain runs until the reply ends (so post-answer checks and
            # revisions are kept) and is graded on its last \\boxed{} (which may span a
            # thought boundary).
            if eos or not o["ids"]:
                if P._last_boxed("".join(s["thoughts"])) is not None:
                    s["status"] = "answer"
                else:
                    s["status"] = "eos_no_answer" if s["thoughts"] else "empty_output"
            elif len(s["ids"]) >= cfg.response_length:
                s["status"] = "budget"
    for s in states:
        s["text"] = "".join(s["thoughts"])
        s["n_tokens"] = len(s["ids"])
        s["n_new_tokens"] = len(s["ids"]) - s["prefix_tokens"]
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
        s["n_new_tokens"] = len(s["gen_ids"])
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


def verify(eng: Engine, cfg, prompt_texts: List[str]) -> List[dict]:
    """Self-verification calls (chat-templated, like localization). An unparseable verdict counts as
    NO (the response continues to localization)."""
    outs = localize(eng, cfg, prompt_texts, max_tokens=cfg.verify_max_tokens)
    for o in outs:
        o["verdict"] = P.parse_verdict(o["response"])
    return outs


def localize(eng: Engine, cfg, prompt_texts: List[str], max_tokens: Optional[int] = None) -> List[dict]:
    """Chat-templated localization calls, no stop sequences."""
    budget = max_tokens if max_tokens is not None else cfg.localization_max_tokens
    ids = [eng.chat_ids(t) for t in prompt_texts]
    params = [sp(cfg, max_tokens=max(1, min(budget, eng.max_model_len - len(i))),
                 temperature=cfg.localization_temp, skip_special_tokens=False) for i in ids]
    outs = eng.generate(ids, params)
    return [{"response": o["text"], "finish_reason": o["finish_reason"],
             "n_tokens": len(o["ids"]), "prompt_tokens": len(i)} for o, i in zip(outs, ids)]


# --- ICS loop -----------------------------------------------------------------------
def is_empty(method: str, g: dict) -> bool:
    return len(g["thoughts"]) == 0 if method == "thought" else not g["text"].strip()


def gen_record(method: str, g: dict, gt: str) -> dict:
    rec = {"n_tokens": g["n_tokens"], "n_new_tokens": g["n_new_tokens"], "gen_status": g["status"],
           "answer": boxed_answer(g["text"]), "correct": is_correct(g["text"], gt)}
    if method == "thought":
        rec.update(chain=g["thoughts"], n_thoughts=len(g["thoughts"]),
                   n_new_thoughts=len(g["thoughts"]) - g["prefix_n"],
                   thought_cap_hits=g["thought_cap_hits"])
    else:
        rec.update(solution=g["text"], finish_reason=g["finish_reason"])
    return rec


def load_shared_init(eng: Engine, thought_records: List[dict], problems: List[dict]) -> Dict[int, dict]:
    """Token-ICS first attempts: the same job's Thought-ICS first attempts, joined into one response.

    Thought chains are generated with the exact tokens so far at every step, so a joined chain is
    distributed exactly as one uninterrupted generation under the same prompt; both methods start
    from the identical first attempt on every problem.
    """
    by_id = {r["problem_id"]: r for r in thought_records}
    out = {}
    for k, p in enumerate(problems):
        r = by_id[p["unique_id"]]
        it0 = r["iterations"][0]
        text = "".join(it0["chain"])
        ids = eng.tok.encode(text, add_special_tokens=False) if text else []
        out[k] = {"text": text, "ids": ids, "n_tokens": len(ids), "n_new_tokens": len(ids),
                  "status": it0["gen_status"], "finish_reason": None,
                  "empty_retries": r.get("empty_retries", 0)}
    return out


def load_thought_init(eng: Engine, path, problems: List[dict]) -> Dict[int, dict]:
    """Thought-ICS first attempts from a previous run (its iteration-0 chains), so that a run
    starts from exactly the same first attempts as the runs it is compared with (``--init-from``).
    Thought texts are identical; token ids are re-encoded per thought."""
    by_id = {json.loads(l)["problem_id"]: json.loads(l) for l in open(Path(path) / "thought_results.jsonl")}
    out = {}
    for k, p in enumerate(problems):
        r = by_id[p["unique_id"]]
        it0 = r["iterations"][0]
        ids, bounds = [], []
        for t in it0["chain"]:
            t_ids = eng.tok.encode(t, add_special_tokens=False)
            bounds.append((len(ids), len(ids) + len(t_ids)))
            ids += t_ids
        out[k] = {"thoughts": list(it0["chain"]), "bounds": bounds, "ids": ids, "text": "".join(it0["chain"]),
                  "n_tokens": len(ids), "n_new_tokens": len(ids), "prefix_n": 0, "status": it0["gen_status"],
                  "thought_cap_hits": it0.get("thought_cap_hits", 0), "empty_retries": r.get("empty_retries", 0)}
    return out


def fit_critique(eng: Engine, cfg, critique: str, builds) -> Tuple[str, int]:
    """Cut the critique from the end until every prompt built from it fits: builds is a list of
    (fn(critique) -> prompt token count, max allowed). Returns (critique, tokens_cut)."""
    toks = eng.tok.encode(critique, add_special_tokens=False)
    cut = 0
    while toks and any(f(eng.tok.decode(toks)) > lim for f, lim in builds):
        over = max(f(eng.tok.decode(toks)) - lim for f, lim in builds)
        n = min(len(toks), max(over, 64))
        toks, cut = toks[:-n], cut + n
    return (eng.tok.decode(toks) if cut else critique), cut


def run_method(eng: Engine, cfg, method: str, problems: List[dict],
               shared_init: Optional[List[dict]] = None) -> List[dict]:
    """Run one method on all problems. Token-ICS (method="token") starts from ``shared_init``, the
    Thought-ICS records of the same job."""
    gen = gen_thought_chains if method == "thought" else gen_token_solutions
    template = P.THOUGHT_TEMPLATE
    prompt_ids = [eng.gen_prompt_ids(template.format(question=p["problem"])) for p in problems]
    records = [{"problem_id": p["unique_id"], "ground_truth": p["answer"],
                "empty_retries": 0, "iterations": []} for k, p in enumerate(problems)]

    # First attempts; a chain that comes back with no thoughts is regenerated (bounded retries).
    current: Dict[int, dict] = {}
    pending = list(range(len(problems)))
    if method == "thought" and cfg.init_from:
        current = load_thought_init(eng, cfg.init_from, problems)
        for i, r in enumerate(records):
            r["empty_retries"] = current[i]["empty_retries"]
        pending = []
    if method == "token":
        current = load_shared_init(eng, shared_init, problems)
        for i, r in enumerate(records):
            r["empty_retries"] = current[i]["empty_retries"]
        pending = []
    for attempt in range(cfg.max_empty_retries + 1):
        if not pending:
            break
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

    SELF = cfg.verify == "self"
    MODE = cfg.feedback if SELF else None
    CARRY = MODE is not None
    critique: Dict[int, str] = {}  # latest verifier critique per problem (replaced each iteration)
    crit_cut: Dict[int, int] = {}
    gen_text = [template.format(question=p["problem"]) for p in problems]
    exit_reason = {}
    for it in range(1, cfg.max_ics_iterations + 1):
        # oracle verification stops correct problems; under self-verification the model decides
        active = [i for i in range(len(records)) if not stopped.get(i) and (SELF or not done[i])]
        if not active:
            break
        if SELF:
            vtexts = [P.verify_prompt(problems[i]["problem"], current[i]["text"], boxed_answer(current[i]["text"]))
                      for i in active]
            verdicts = verify(eng, cfg, vtexts)
            still = []
            for i, vt, v in zip(active, vtexts, verdicts):
                vinfo = {"verify_prompt": vt, **v}
                if v["verdict"] is True:  # verified as correct: stop
                    stopped[i] = True
                    exit_reason[i] = "verified"
                    records[i]["iterations"].append({"iteration": it, "verification": vinfo, "stopped": "verified"})
                else:
                    still.append(i)
                    records[i].setdefault("_pending_verification", {})[it] = vinfo
                    if CARRY:
                        critique[i] = response_text(v["response"]).strip()
            active = still
            if not active:
                break
        def loc_text(i, c=None):
            if method == "thought":
                return P.thought_localization_prompt(problems[i]["problem"], current[i]["thoughts"],
                                                     cfg.verify, critique=c)
            return P.token_localization_prompt(problems[i]["problem"], current[i]["text"], cfg.verify, critique=c)
        if CARRY:
            for i in active:
                # localizer keeps its full budget; the resample prompt leaves the full generation budget
                fits = [(lambda c, i=i: len(eng.chat_ids(loc_text(i, c))), eng.max_model_len - cfg.localization_max_tokens)]
                if MODE == "all":
                    fits.append((lambda c, i=i: len(eng.gen_prompt_ids(P.feedback_gen_prompt(gen_text[i], c))),
                                 eng.max_model_len - cfg.response_length))
                critique[i], crit_cut[i] = fit_critique(eng, cfg, critique[i], fits)
        texts = [loc_text(i, critique[i] if CARRY else None) for i in active]
        locs = localize(eng, cfg, texts)

        jobs, job_idx, loc_info, failed = [], [], {}, {}
        loc_msg: Dict[int, str] = {}

        def resample_ids(i):
            if MODE != "all":
                return prompt_ids[i]
            # the localizer's reply goes in after the (already fitted) critique; cut its tail if needed
            m, c = fit_critique(eng, cfg, loc_msg[i], [(lambda x: len(eng.gen_prompt_ids(
                P.feedback_gen_prompt(gen_text[i], critique[i], x))), eng.max_model_len - cfg.response_length)])
            loc_info[i]["resample_localization"], loc_info[i]["localization_tokens_cut"] = m, c
            return eng.gen_prompt_ids(P.feedback_gen_prompt(gen_text[i], critique[i], m))
        for i, text, loc in zip(active, texts, locs):
            g = current[i]
            info = {"loc_prompt": text, **loc}
            loc_info[i] = info
            if MODE == "all":
                loc_msg[i] = response_text(loc["response"]).strip()
            if CARRY:
                info.update(critique=critique[i], critique_tokens_cut=crit_cut.get(i, 0))
            if method == "thought":
                n = len(g["thoughts"])
                if loc["n_tokens"] == 0:
                    step, status = None, "empty"
                else:
                    step, status = P.parse_error_step(loc["response"], n)
                info.update(parse_status=status, error_step=step)
                if status in ("default", "empty"):
                    failed[i] = info  # no usable localization: the attempt is used up
                    continue
                if step == 0:
                    stopped[i] = True
                    loc_info[i] = info
                    continue
                keep = step - 1
                cut = g["bounds"][keep - 1][1] if keep > 0 else 0
                job = {"prompt_ids": resample_ids(i), "prefix_ids": g["ids"][:cut],
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
                info["quote_found"] = idx is not None
                if idx is None:
                    failed[i] = info  # no usable localization (unparseable / not found)
                    continue
                prefix_text = g["text"][:idx].rstrip() if idx else ""
                job = {"prompt_ids": resample_ids(i), "prefix_text": prefix_text,
                       "prefix_ids": eng.tok.encode(prefix_text, add_special_tokens=False)
                       if prefix_text else []}
                info.update(truncation_idx=idx, prefix_len=len(job["prefix_ids"]))
            jobs.append(job)
            job_idx.append(i)
            loc_info[i] = info

        for i in active:
            if SELF:
                (failed[i] if i in failed else loc_info[i])["verification"] = \
                    records[i]["_pending_verification"].pop(it)
            if stopped.get(i):
                exit_reason[i] = "vl_disagreement"  # verifier said wrong, localizer found no error
                records[i]["iterations"].append({"iteration": it, "localization": loc_info[i],
                                                 "stopped": "no_error_found"})
            elif i in failed:
                # attempt used up; the next iteration re-localizes the same chain
                records[i]["iterations"].append({"iteration": it, "localization": failed[i],
                                                 "failed_localization": True})
        for i, g in zip(job_idx, gen(eng, cfg, jobs) if jobs else []):
            rec = gen_record(method, g, records[i]["ground_truth"])
            rec.update(iteration=it, localization=loc_info[i])
            records[i]["iterations"].append(rec)
            current[i] = g
            done[i] = rec["correct"]
        logger.info(f"[{method}] iter {it}: accuracy {sum(done.values())}/{len(records)}")

    for i, r in enumerate(records):
        r.pop("_pending_verification", None)
        if SELF:
            # final answer = the last generated solution
            last = [it for it in r["iterations"] if "gen_status" in it][-1]
            r["exit"] = exit_reason.get(i, "max_iter")
            r["final_correct"] = last["correct"]
        else:
            r["final_correct"] = done[i]
    return records


# --- Summary ------------------------------------------------------------------------
def summarize(method: str, records: List[dict]) -> dict:
    n = len(records)
    gens = [it for r in records for it in r["iterations"] if "gen_status" in it]
    locs = [it["localization"] for r in records for it in r["iterations"] if "localization" in it]
    init = sum(r["iterations"][0]["correct"] for r in records)
    final = sum(r["final_correct"] for r in records)
    sv = {}
    if records and "exit" in records[0]:
        vs = [it["verification"] if "verification" in it else it.get("localization", {}).get("verification")
              for r in records for it in r["iterations"][1:]]
        sv = {"exits": dict(Counter(r["exit"] for r in records)),
              "verifications": sum(v is not None for v in vs),
              "verifier_unparseable": sum(v is not None and v["verdict"] is None for v in vs)}
    budget_status = ("budget", "max_thoughts", "context")
    s = {
        "method": method, "n_problems": n,
        "init_accuracy": init / n, "final_accuracy": final / n, "lift": (final - init) / n,
        "initial_gen_status": dict(Counter(r["iterations"][0]["gen_status"] for r in records)),
        "initial_no_answer_from_budget": sum(r["iterations"][0]["gen_status"] in budget_status
                                             and r["iterations"][0]["answer"] == "NO ANSWER"
                                             for r in records) / n,
        # compute actually spent (generation incl. resamples, and localization)
        "total_new_gen_tokens": sum(g["n_new_tokens"] for g in gens),
        "total_loc_tokens": sum(l["n_tokens"] for l in locs),
        "all_gen_status": dict(Counter(g["gen_status"] for g in gens)),
        "empty_retries": sum(r["empty_retries"] for r in records),
        "n_localizations": len(locs),
        "loc_finish_length_rate": (sum(l["finish_reason"] == "length" for l in locs) / len(locs))
        if locs else None,
        "loc_parse_status": dict(Counter(l["parse_status"] for l in locs)),
        "failed_localizations": sum(it.get("failed_localization", False)
                                    for r in records for it in r["iterations"]),
        # resamples that start from an empty prefix because the model localized the very start
        "resamples_from_start": sum(it["localization"].get("prefix_len", 1) == 0
                                    for r in records for it in r["iterations"]
                                    if "gen_status" in it and "localization" in it),
        "resamples": sum("gen_status" in it for r in records for it in r["iterations"][1:]),
        **sv,
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
    ap.add_argument("--dataset", required=True, choices=BENCHMARKS)
    ap.add_argument("--method", choices=["both", "thought"], default="both",
                    help="both = Thought-ICS, then Token-ICS from the same first attempts")
    ap.add_argument("--verify", choices=["oracle", "self"], default="oracle")
    ap.add_argument("--feedback", choices=["loc", "all"], default=None,
                    help="with --verify self: loc = verification reasoning to the localizer; all = also "
                         "both reasonings to the resampling step (Thought-ICS-F)")
    ap.add_argument("--init-from", default=None,
                    help="run directory whose Thought-ICS first attempts this run starts from")
    ap.add_argument("--max-ics-iterations", type=int, default=P.MAX_ICS_ITERATIONS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems (smoke tests)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results")
    cfg = ap.parse_args()
    if cfg.feedback and cfg.verify != "self":
        ap.error("--feedback requires --verify self")
    for k in ("response_length", "localization_max_tokens", "verify_max_tokens", "max_tokens_per_thought",
              "max_thoughts", "max_empty_retries", "generation_temp", "localization_temp", "top_p", "top_k"):
        setattr(cfg, k, getattr(P, k.upper()))

    m = MODELS[cfg.model]
    problems = load_benchmark(cfg.dataset)[:cfg.limit]
    run_dir = Path(cfg.out_dir) / (f"{cfg.model}_{run_tag(cfg.dataset)}_{MODE_TAG[(cfg.verify, cfg.feedback)]}"
                                   f"_{datetime.now():%Y%m%d_%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                         text=True).strip()
    except Exception:
        commit = None
    config = {**vars(cfg), "hf": m["hf"], "tp": m["tp"], "max_model_len": P.MAX_MODEL_LEN,
              "final_prefill": m["final_prefill"], "prefix_caching": m.get("prefix_caching", True),
              "git_commit": commit}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info(f"Config: {json.dumps(config)}")

    eng = Engine(m["hf"], m["tp"], P.MAX_MODEL_LEN, cfg.seed, cfg.gpu_mem, m["final_prefill"],
                 m.get("prefix_caching", True))
    methods = ["thought", "token"] if cfg.method == "both" else ["thought"]
    summaries, thought_records = {}, None
    for method in methods:
        t0 = time.time()
        records = run_method(eng, cfg, method, problems, shared_init=thought_records if method == "token" else None)
        if method == "thought":
            thought_records = records
        with open(run_dir / f"{method}_results.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        summaries[method] = {**summarize(method, records), "wall_seconds": time.time() - t0}
        logger.info(f"[{method}] summary: {json.dumps(summaries[method])}")
        (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    logger.info(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
