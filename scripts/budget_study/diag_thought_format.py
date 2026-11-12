"""Thought-format compliance under prompt / stop-sequence variants (chat template, final
channel for gpt-oss). Measures how often the model ends thoughts with </thought> itself.

    python scripts/budget_study/diag_thought_format.py <model_key> [n_per_dataset]
"""
import statistics as st
import sys

from vllm import SamplingParams

from thought_ics.budget_study import protocol as P
from thought_ics.budget_study import run_study as R
from thought_ics.datasets import load_dataset_by_name
from thought_ics.paper_prompts import GENERATION_PROMPT as PAPER_TEMPLATE  # 2 worked examples

model = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
m = R.MODELS[model]
eng = R.Engine(m["hf"], m["tp"], m["max_model_len"], 42, 0.85, m["final_prefill"],
               m.get("prefix_caching", True))
probs = (load_dataset_by_name("aime", n_problems=n, seed=42)
         + load_dataset_by_name("math500", n_problems=n, level=5, seed=42))

VARIANTS = {
    "srpo_noex__delim": (P.THOUGHT_TEMPLATE, [P.THOUGHT_DELIMITER]),
    "paper_ex__delim": (PAPER_TEMPLATE, [P.THOUGHT_DELIMITER]),
    "srpo_noex__delim+nn": (P.THOUGHT_TEMPLATE, [P.THOUGHT_DELIMITER, "\n\n"]),
    "paper_ex__delim+nn": (PAPER_TEMPLATE, [P.THOUGHT_DELIMITER, "\n\n"]),
}
CAP, MAX_THOUGHTS, TOTAL = 2048, 20, 16384

for name, (tmpl, stop) in VARIANTS.items():
    states = [{"ids": eng.gen_prompt_ids(tmpl.format(question=p["problem"])), "n": 0, "done": None,
               "lens": [], "ends": []} for p in probs]
    gen_len = [0] * len(states)
    for _ in range(MAX_THOUGHTS):
        act = [k for k, s in enumerate(states) if s["done"] is None and gen_len[k] + CAP <= TOTAL]
        if not act:
            break
        sp = SamplingParams(temperature=0.5, top_p=0.9, top_k=50, max_tokens=CAP, stop=stop,
                            include_stop_str_in_output=True, skip_special_tokens=True)
        outs = eng.generate([states[k]["ids"] for k in act], [sp] * len(act))
        for k, o in zip(act, outs):
            s = states[k]
            if not o["ids"]:
                s["done"] = "empty"
                continue
            s["ids"] = s["ids"] + o["ids"]
            gen_len[k] += len(o["ids"])
            s["n"] += 1
            s["lens"].append(len(o["ids"]))
            s["ends"].append(o["stop_reason"] if o["stop_reason"] is not None else o["finish_reason"])
            if "\\boxed{" in o["text"]:
                s["done"] = "answer"
            elif o["finish_reason"] == "stop" and o["stop_reason"] is None:
                s["done"] = "eos"
    ends = [e for s in states for e in s["ends"]]
    lens = [x for s in states for x in s["lens"]]
    print(f"RESULT {model} {name}: thoughts/chain med={st.median(s['n'] for s in states)} "
          f"| ends: delim={ends.count(P.THOUGHT_DELIMITER)/len(ends):.2f} "
          f"nn={ends.count(chr(10)*2)/len(ends):.2f} cap={ends.count('length')/len(ends):.2f} "
          f"eos={sum(e == 'stop' for e in ends)/len(ends):.2f} "
          f"| thought tok med={st.median(lens)} p90={sorted(lens)[int(.9*len(lens))]} "
          f"| answered={sum(s['done']=='answer' for s in states)}/{len(states)}", flush=True)
