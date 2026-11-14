"""Self-Refine and Chain-of-Verification (CoVe) baselines.

Both start from the same initial response as Thought-ICS (its thought-by-thought generation, joined
into one response), use the same engine, chat templates, sampling settings and per-call budget
(16,384 tokens, capped by the remaining context), and are graded the same way.

- Self-Refine (Madaan et al., 2023), with generic feedback prompts following Huang et al. (2024):
  feedback -> stop if it says YES -> refine from scratch with the feedback, for up to 10
  iterations (the number of correction attempts Thought-ICS gets).
- CoVe (Dhuliawala et al., 2023), factored variant: plan up to 4 verification questions, answer
  each without the solution in context, then write a final solution. One pass, by design.

Text carried into a later prompt (Self-Refine feedback, CoVe verification answers) is the model's
whole response; for gpt-oss that is its analysis reasoning followed by its final message, as for
Thought-ICS-F. Verdicts and question lists are parsed from the final message.

    python -m thought_ics.baselines --model gptoss20b --dataset aime --init-from <self-verified run dir>
"""
import argparse
import json
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from thought_ics import protocol as P
from thought_ics.benchmarks import BENCHMARKS, load_benchmark, run_tag
from thought_ics.ics import MODELS, Engine, boxed_answer, final_message, is_correct, response_text, sp

# --- Self-Refine prompts and feedback parser (verbatim) --------------------------------
FEEDBACK_PROMPT = """You are reviewing a solution to a math problem. Analyze it carefully for errors.

Problem: {problem}

Solution to review:
{solution}

Provide feedback on this solution:
1. Is the solution correct? Answer YES or NO at the start.
2. If NO, identify the specific error(s) and explain what went wrong.
3. If YES, confirm the solution is correct.

Feedback:"""

REFINE_PROMPT = """You previously attempted to solve a problem but received feedback indicating errors. Use the feedback to produce a corrected solution.

Problem: {problem}

Your previous solution:
{solution}

Feedback on your solution:
{feedback}

Now provide a corrected solution. Show your reasoning step by step and put your final answer in \\boxed{{}}.

Corrected Solution:"""


def parse_feedback(feedback: str) -> Tuple[bool, str]:
    """Parse feedback to determine if solution is deemed correct.

    Returns:
        Tuple of (is_correct, reasoning)
    """
    feedback_lower = feedback.lower().strip()

    # Check for explicit YES/NO at the start
    if feedback_lower.startswith("yes"):
        return True, "Feedback indicates solution is correct"
    if feedback_lower.startswith("no"):
        return False, "Feedback indicates errors in solution"

    # Check for other positive indicators
    positive_patterns = [
        r"the solution is correct",
        r"this is correct",
        r"the answer is correct",
        r"correct solution",
        r"no errors",
        r"no issues",
        r"well done",
        r"correct!",
    ]

    for pattern in positive_patterns:
        if re.search(pattern, feedback_lower):
            return True, f"Matched positive pattern: {pattern}"

    # Check for negative indicators
    negative_patterns = [
        r"incorrect",
        r"error",
        r"mistake",
        r"wrong",
        r"should be",
        r"instead of",
        r"fix",
        r"correct this",
    ]

    for pattern in negative_patterns:
        if re.search(pattern, feedback_lower):
            return False, f"Matched negative pattern: {pattern}"

    # Default to assuming refinement is needed if unclear
    return False, "Unclear feedback, assuming refinement needed"


# --- CoVe prompts and question parser (verbatim) -----------------------------------
PLAN_VERIFICATION_PROMPT = """You have provided a solution to a math problem. Now generate verification questions to check if your solution is correct.

Problem: {problem}

Your solution:
{baseline}

Generate 2-4 specific verification questions that would help verify the correctness of this solution. Each question should check a specific step, calculation, or reasoning in the solution.

Format your questions as a numbered list:
1. [First verification question]
2. [Second verification question]
...

Verification Questions:"""

EXECUTE_VERIFICATION_PROMPT = """Answer the following question carefully and precisely.

Question: {question}

Answer:"""

FINAL_RESPONSE_PROMPT = """You previously solved a problem and generated verification questions with answers. Use this verified information to produce a final, corrected solution.

Problem: {problem}

Your baseline solution:
{baseline}

Verification questions and answers:
{verifications}

Based on the verification results, provide your final solution. If the verifications revealed any errors, correct them. Show your reasoning step by step and put your final answer in \\boxed{{}}.

Final Solution:"""


def parse_verification_questions(text: str) -> List[str]:
    """Parse numbered list of verification questions from model output.

    Args:
        text: Model output containing numbered questions

    Returns:
        List of question strings
    """
    questions = []

    # Try to match numbered patterns like "1.", "1)", "1:", etc.
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        # Match patterns like "1. question" or "1) question" or "- question"
        match = re.match(r'^(?:\d+[.\):]|\-|\*)\s*(.+)$', line)
        if match:
            question = match.group(1).strip()
            if question and len(question) > 5:  # Filter out very short lines
                questions.append(question)

    # If no numbered list found, try to split by double newlines
    if not questions:
        paragraphs = text.strip().split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if para and '?' in para:
                questions.append(para)

    # Limit to 4 questions max
    return questions[:4]


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def feedback_verdict(text: str):
    """The paper's Self-Refine parser (parse_feedback, verbatim), applied after removing the
    formatting around the leading YES/NO that the prompt asks for: markdown emphasis/headers
    ("**NO**", "### YES") and a leading "Answer:" / "1." label. Without this, "**Answer: NO** ...
    fundamental mistakes" never hits the startswith check and falls through to the phrase list
    (which matched "correct solution" -> YES)."""
    t = re.sub(r"[*_#`>]+", "", text).strip()
    t = re.sub(r"^(?:\d+[.):]\s*|answer\s*[:\-]\s*|feedback\s*[:\-]\s*)+", "", t, flags=re.I).strip()
    return parse_feedback(t)


MIN_OUTPUT_ROOM = 4096


def fit(eng, build, parts):
    """build(parts) -> prompt text. If the solution-writing prompt would leave less than
    MIN_OUTPUT_ROOM tokens of context, cut the fed-back evaluation texts in `parts` (Self-Refine
    feedback / CoVe verification questions and answers) from the end, longest first, until it fits. Only
    happens when an evaluation looped to its budget (<3% of calls); the solution and problem
    are never cut. Returns (prompt, tokens_cut)."""
    limit = eng.max_model_len - MIN_OUTPUT_ROOM
    toks = [eng.tok.encode(t, add_special_tokens=False) for t in parts]
    cut = 0
    while len(eng.gen_prompt_ids(build([eng.tok.decode(t) for t in toks]))) > limit:
        j = max(range(len(toks)), key=lambda i: len(toks[i]))
        if not toks[j]:
            break
        over = len(eng.gen_prompt_ids(build([eng.tok.decode(t) for t in toks]))) - limit
        n = min(len(toks[j]), max(over, 64))
        toks[j] = toks[j][:-n]
        cut += n
    return build([eng.tok.decode(t) for t in toks]) if cut else build(parts), cut


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
        # "text": the final message (parsed for verdicts / question lists); "full": the whole response
        # (gpt-oss: analysis reasoning + final message), which is what gets carried into later prompts,
        # as for Thought-ICS-F. Identical for models without a separate reasoning channel.
        res.append({"raw": o["text"], "text": msg.strip(), "full": (msg if solution else response_text(o["text"])).strip(),
                    "finish_reason": o["finish_reason"],
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
        fbs = call(eng, cfg, [FEEDBACK_PROMPT.format(problem=problems[i]["problem"], solution=cur[i])
                              for i in active], solution=False)
        refine = []
        for i, fb in zip(active, fbs):
            ok, why = feedback_verdict(fb["text"])
            recs[i]["iterations"][-1]["feedback"] = {**fb, "says_correct": ok, "parse": why}
            if ok:
                recs[i]["stopped_early"] = True
            else:
                refine.append((i, fb["full"]))
        built = [fit(eng, lambda ps, i=i: REFINE_PROMPT.format(problem=problems[i]["problem"], solution=cur[i],
                                                                  feedback=ps[0]), [f]) for i, f in refine]
        for (i, _), (_, c) in zip(refine, built):
            if c:
                recs[i]["iterations"][-1]["feedback"]["tokens_cut_to_fit"] = c
        outs = call(eng, cfg, [b for b, _ in built], solution=True)
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
    plans = call(eng, cfg, [PLAN_VERIFICATION_PROMPT.format(problem=p["problem"], baseline=init[k])
                            for k, p in enumerate(problems)], solution=False)
    qs = [parse_verification_questions(pl["text"])[:4] for pl in plans]
    flat = [(k, q) for k in range(n) for q in qs[k]]
    ans = call(eng, cfg, [EXECUTE_VERIFICATION_PROMPT.format(question=q) for _, q in flat], solution=False)
    by = [[] for _ in range(n)]
    for (k, q), a in zip(flat, ans):
        by[k].append((q, a))
    finals, cut_k = [], {}
    for k, p in enumerate(problems):
        nq = len(by[k])

        def build(parts, k=k, p=p, nq=nq):  # parts = questions + answers (both model-written eval text)
            vt = "".join(f"Q{i}: {q}\nA{i}: {a}\n\n" for i, (q, a) in enumerate(zip(parts[:nq], parts[nq:]), 1))
            if not vt.strip():
                vt = "(No verification questions generated)"
            return FINAL_RESPONSE_PROMPT.format(problem=p["problem"], baseline=init[k], verifications=vt)
        prompt, c = fit(eng, build, [q for q, _ in by[k]] + [a["full"] for _, a in by[k]])
        finals.append(prompt)
        cut_k[k] = c
    outs = call(eng, cfg, finals, solution=True)
    recs = []
    for k, p in enumerate(problems):
        recs.append({"problem_id": p["unique_id"], "ground_truth": p["answer"],
                     "iterations": [sol_record(init[k], p["answer"], iteration=0),
                                    sol_record(outs[k]["text"], p["answer"], iteration=1,
                                               finish_reason=outs[k]["finish_reason"], n_tokens=outs[k]["n_tokens"])],
                     "plan": plans[k], "verifications": [{"question": q, **a} for q, a in by[k]],
                     "tokens_cut_to_fit": cut_k[k]})
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--dataset", required=True, choices=BENCHMARKS)
    ap.add_argument("--method", choices=["self_refine", "cove", "both"], default="both")
    ap.add_argument("--init-from", default=None,
                    help="self-verified run directory whose Thought-ICS first attempts the baselines start "
                         "from (default: the latest matching run in --out-dir)")
    ap.add_argument("--max-iterations", type=int, default=P.MAX_ICS_ITERATIONS, help="Self-Refine iterations")
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems (smoke tests)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results")
    cfg = ap.parse_args()
    cfg.budget, cfg.temp, cfg.top_p, cfg.top_k = P.RESPONSE_LENGTH, P.GENERATION_TEMP, P.TOP_P, P.TOP_K
    m = MODELS[cfg.model]
    tag = run_tag(cfg.dataset)
    if cfg.init_from is None:
        cands = sorted(d for d in Path(cfg.out_dir).glob(f"{cfg.model}_{tag}_self_2*")
                       if (d / "thought_results.jsonl").exists())
        if not cands:
            raise SystemExit(f"no finished self-verified run for {cfg.model} {tag} in {cfg.out_dir}")
        cfg.init_from = str(cands[-1])
    problems = load_benchmark(cfg.dataset)[:cfg.limit]
    init = load_init(Path(cfg.init_from), problems)
    run_dir = Path(cfg.out_dir) / f"{cfg.model}_{tag}_baselines_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        commit = None
    config = {**vars(cfg), "hf": m["hf"], "tp": m["tp"], "max_model_len": P.MAX_MODEL_LEN,
              "final_prefill": m["final_prefill"], "git_commit": commit}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info(f"Config: {json.dumps(config)}")
    eng = Engine(m["hf"], m["tp"], P.MAX_MODEL_LEN, cfg.seed, cfg.gpu_mem, m["final_prefill"],
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
