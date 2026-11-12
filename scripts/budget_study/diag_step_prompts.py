"""Generate first-attempt Thought chains under step-prompt variants (for judging atomicity).

    python scripts/budget_study/diag_step_prompts.py <model_key> <out_dir> [n_per_dataset]

Writes <out_dir>/<variant>/<model>.jsonl with question, ground truth, chain (live boundary
rules), status, correctness. Judge afterwards with judge_thoughts.py --variants <out_dir>.
"""
import json
import sys
import types
from pathlib import Path

from thought_ics.budget_study import protocol as P
from thought_ics.budget_study import run_study as R
from thought_ics.datasets import load_dataset_by_name

RULES = """Keep each step to a single paragraph (no blank lines inside a step), and do not insert separator lines (such as ---) between steps. Continue until you reach the final answer, and include it as \\boxed{{answer}} in the step that arrives at it."""

V1 = """Solve the problem step by step. Write your reasoning as a sequence of steps, one step per paragraph, with a blank line between steps.

Each step must do exactly one thing: a single observation, calculation, or deduction, usually one or two sentences. Whenever you move on to something new (a new calculation, a new case, a new deduction, the next option), start a new paragraph. Each step should:
- Make forward progress toward the solution
- Contain substantive reasoning (not filler like "let me think" or restating the problem)
- Cohere logically with the previous steps

Keep each step to a single paragraph (no blank lines inside a step), keep any equation in the same paragraph as the sentence that introduces it, and do not insert headers or separator lines (such as ---) between steps. Continue until you reach the final answer, and include it as \\boxed{{answer}} in the step that arrives at it.

Q: {question}
"""

EXAMPLES = """Examples:

Q: In how many ways can 5 distinct books be arranged on a shelf if 2 specific books must not be adjacent?

Total arrangements without restrictions is 5! = 120.

I need to subtract the arrangements where the 2 specific books are adjacent.

If I treat the 2 books as a single unit, I have 4 units to arrange: 4! = 24 ways.

The 2 books within their unit can be arranged in 2! = 2 ways.

So the arrangements with the books adjacent number 24 × 2 = 48.

Therefore, the arrangements where they are not adjacent number 120 - 48 = \\boxed{{72}}.

Q: A rectangle has area 48 and perimeter 28. What is the length of its diagonal?

Let length = l and width = w. From the area: lw = 48.

From the perimeter: 2l + 2w = 28, so l + w = 14.

From l + w = 14, we get w = 14 - l. Substituting into lw = 48 gives l(14 - l) = 48.

Expanding: 14l - l² = 48, so l² - 14l + 48 = 0, which factors as (l - 6)(l - 8) = 0.

So l = 8 and w = 6 (or vice versa). By the Pythagorean theorem, d² = 8² + 6² = 64 + 36 = 100.

Therefore d = 10, so the answer is \\boxed{{10}}.

"""
V2 = V1.replace("Q: {question}\n", EXAMPLES + "Q: {question}\n")
VARIANTS = {"V0_current": P.THOUGHT_TEMPLATE, "V1_tight": V1, "V2_tight_examples": V2}

model, out_dir = sys.argv[1], Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
m = R.MODELS[model]
eng = R.Engine(m["hf"], m["tp"], m["max_model_len"], 42, 0.85, m["final_prefill"], m.get("prefix_caching", True))
probs = []
for ds, lvl in (("amc23", None), ("aime", None), ("math500", 5), ("csqa", None), ("gpqa", None), ("mathqa", None)):
    probs += [dict(p, dataset=ds) for p in load_dataset_by_name(ds, n_problems=n, level=lvl, seed=7)]
cfg = types.SimpleNamespace(max_thoughts=P.MAX_THOUGHTS, max_tokens_per_thought=P.MAX_TOKENS_PER_THOUGHT,
                            response_length=m["response_length"], generation_temp=P.GENERATION_TEMP,
                            top_p=P.TOP_P, top_k=P.TOP_K)
for name, tmpl in VARIANTS.items():
    jobs = [{"prompt_ids": eng.gen_prompt_ids(tmpl.format(question=p["problem"]))} for p in probs]
    chains = R.gen_thought_chains(eng, cfg, jobs)
    (out_dir / name).mkdir(parents=True, exist_ok=True)
    with open(out_dir / name / f"{model}.jsonl", "w") as f:
        for p, g in zip(probs, chains):
            f.write(json.dumps({"model": model, "variant": name, "dataset": p["dataset"],
                                "problem_id": p.get("unique_id"), "question": p["problem"],
                                "ground_truth": p["answer"], "chain": g["thoughts"],
                                "status": g["status"], "n_tokens": g["n_tokens"],
                                "correct": R.is_correct(g["text"], p["answer"])}) + "\n")
    ok = sum(R.is_correct(g["text"], p["answer"]) for p, g in zip(probs, chains))
    ans = sum(g["status"] == "answer" for g in chains)
    print(f"RESULT {model} {name}: correct {ok}/{len(probs)} answered {ans}/{len(probs)} "
          f"steps med {sorted(len(g['thoughts']) for g in chains)[len(chains)//2]}", flush=True)
