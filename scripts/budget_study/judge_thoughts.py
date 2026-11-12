"""Judge Thought-ICS thought quality with an LLM judge (default gpt-oss-120b, local vLLM).

Rubric = SRPO's thought_evaluator dimensions (forward progress, substantiveness, coherence;
1-5) plus an atomicity label: ATOMIC (exactly one complete, self-contained reasoning step),
FRAGMENT (incomplete on its own: a lead-in, a bare result, a dangling clause) or BUNDLED
(several distinct steps). Thoughts are judged under the old boundaries (one paragraph = one
thought) and the current boundary rules (protocol.has_content / is_display_math / opens_next),
which depend only on the current and previous paragraph, so they reproduce the inference-time
boundaries exactly.

    python scripts/budget_study/judge_thoughts.py <results_root> <out.json> [chains_per_model]
"""
import glob
import json
import random
import re
import sys
from collections import Counter, defaultdict

from thought_ics.budget_study import protocol as P
from thought_ics.budget_study import run_study as R

root, out_path = sys.argv[1], sys.argv[2]
per_model = int(sys.argv[3]) if len(sys.argv) > 3 else 30
MAX_THOUGHTS_JUDGED = 12
random.seed(0)


def resegment(chain):
    out = []
    for t in chain:
        if out and (not P.has_content(t) or P.is_display_math(t) or P.opens_next(out[-1])):
            out[-1] += t
        else:
            out.append(t)
    return out


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


items = []
for model in ("qwen32b", "gptoss20b", "gptoss120b"):
    runs = [d for d in glob.glob(f"{root}/{model}_*_L2_2*/") if "shared" not in d]
    chains = []
    for d in runs:
        for line in open(d + "thought_results.jsonl"):
            r = json.loads(line)
            chains.append((r["problem_id"], d, r["iterations"][0]["chain"]))
    for pid, d, chain in random.sample(chains, min(per_model, len(chains))):
        for seg_name, seg in (("paragraph", [t for t in chain if t.strip()]), ("rules", resegment(chain))):
            for k, step in enumerate(seg[:MAX_THOUGHTS_JUDGED]):
                items.append({"model": model, "segmentation": seg_name, "problem_id": pid, "run": d,
                              "k": k, "history": seg[:k], "step": step})

# questions: the problem text is in the localization prompt only for wrong answers, so reload
from thought_ics.datasets import load_dataset_by_name  # noqa: E402
qcache = {}
for it in items:
    if it["run"] not in qcache:
        c = json.load(open(it["run"] + "config.json"))
        probs = load_dataset_by_name(c["dataset"], n_problems=c["n_problems"], level=c.get("level"), seed=42)
        qcache[it["run"]] = {p.get("unique_id"): p["problem"] for p in probs}
    it["question"] = qcache[it["run"]][it["problem_id"]]

m = R.MODELS["gptoss120b"]
eng = R.Engine(m["hf"], m["tp"], m["max_model_len"], 42, 0.85, m["final_prefill"], m.get("prefix_caching", True))
from vllm import SamplingParams  # noqa: E402
sp = SamplingParams(temperature=0.0, max_tokens=2048, skip_special_tokens=False)
prompts = []
for it in items:
    hist = "".join(f"Step {i}: {t.strip()}\n" for i, t in enumerate(it["history"], 1)) or "(none: this is the first step)\n"
    prompts.append(eng.chat_ids(PROMPT.format(question=it["question"], history=hist, step=it["step"].strip())))
outs = eng.generate(prompts, [sp] * len(prompts))
for it, o in zip(items, outs):
    it.update(parse(o["text"]))
    del it["history"]

summary = {}
groups = defaultdict(list)
for it in items:
    groups[(it["model"], it["segmentation"])].append(it)
for (model, seg), xs in sorted(groups.items()):
    ok = [x for x in xs if x["atomicity"]]
    lab = Counter(x["atomicity"] for x in ok)
    mean = lambda k: sum(x[k] for x in xs if x[k]) / max(1, sum(1 for x in xs if x[k]))
    summary[f"{model}/{seg}"] = {
        "n": len(xs), "parsed": len(ok),
        **{k: round(mean(k), 2) for k in ("forward_progress", "substantiveness", "coherence")},
        **{f"%{k}": round(100 * lab[k] / max(1, len(ok)), 1) for k in ("ATOMIC", "FRAGMENT", "BUNDLED")},
    }
    print(f"RESULT {model:11s} {seg:9s} " + json.dumps(summary[f"{model}/{seg}"]), flush=True)
json.dump({"summary": summary, "items": items}, open(out_path, "w"))
