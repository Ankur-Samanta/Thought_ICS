"""Thought-quality judge rubric (SRPO thought_evaluator dimensions + atomicity label)."""
import re

PROMPT = """You are a HARSH critic evaluating ONE step of a step-by-step solution. Do NOT solve the problem. Only rate the step.

## Problem
{question}

## Previous steps
{history}

## Step to evaluate
{step}

## Rubric (be strict)
forward_progress (1-5): 1 = restating / repeating / circular; 3 = some progress; 5 = major advance.
substantiveness (1-5): 1 = pure filler ("let me think", announcing what comes next); 3 = mixed; 5 = concrete reasoning or calculation.
coherence (1-5): 1 = contradicts or is disconnected from the previous steps; 3 = loosely connected; 5 = perfect continuation.
atomicity (label): ATOMIC = exactly one complete, self-contained reasoning step (one observation, calculation or deduction that stands on its own); FRAGMENT = incomplete on its own (e.g. a lead-in such as "Subtracting the equations gives" without its result, a bare equation or result without the step that produces it, a dangling clause); BUNDLED = several distinct reasoning steps packed into one.

Reply with EXACTLY four lines:
forward_progress: N
substantiveness: N
coherence: N
atomicity: ATOMIC|FRAGMENT|BUNDLED
"""


def parse(resp):
    tail = resp.split("<|channel|>final<|message|>")[-1]
    out = {}
    for k in ("forward_progress", "substantiveness", "coherence"):
        m = re.findall(rf"{k}\s*:\s*([1-5])", tail)
        out[k] = int(m[-1]) if m else None
    m = re.findall(r"atomicity\s*:\s*(ATOMIC|FRAGMENT|BUNDLED)", tail, re.I)
    out["atomicity"] = m[-1].upper() if m else None
    return out




def judge(eng, items, max_tokens=2048):
    """items: dicts with question, history (list of thoughts), step. Adds parsed scores."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, skip_special_tokens=False)
    prompts = []
    for it in items:
        hist = "".join(f"Step {i}: {t.strip()}\n" for i, t in enumerate(it["history"], 1)) \
            or "(none: this is the first step)\n"
        prompts.append(eng.chat_ids(PROMPT.format(question=it["question"], history=hist,
                                                  step=it["step"].strip())))
    for it, o in zip(items, eng.generate(prompts, [sp] * len(prompts))):
        it.update(parse(o["text"]))
    return items
