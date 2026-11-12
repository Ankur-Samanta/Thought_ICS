"""Thought structure under chat format: thoughts inside one assistant reply (current) vs one
assistant turn per thought, separated by a fixed user message.

    python scripts/budget_study/diag_thought_turns.py <model_key> [n_per_dataset]
"""
import statistics as st
import sys

from vllm import SamplingParams

from thought_ics.budget_study import protocol as P
from thought_ics.budget_study import run_study as R
from thought_ics.datasets import load_dataset_by_name

model = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
m = R.MODELS[model]
eng = R.Engine(m["hf"], m["tp"], m["max_model_len"], 42, 0.85, m["final_prefill"],
               m.get("prefix_caching", True))
probs = (load_dataset_by_name("aime", n_problems=n, seed=42)
         + load_dataset_by_name("math500", n_problems=n, level=5, seed=42))
CAP, MAX_THOUGHTS, TOTAL = 2048, 20, 16384
NEXT = "Next step."
header = eng.tok.encode(R.HARMONY_FINAL_HEADER, add_special_tokens=False) if m["final_prefill"] else []


def chat(messages):
    ids = eng.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids) + header


def report(name, states):
    ends = [e for s in states for e in s["ends"]]
    lens = [x for s in states for x in s["lens"]]
    print(f"RESULT {model} {name}: thoughts/chain med={st.median(s['n'] for s in states)} "
          f"| ends: delim={ends.count(P.THOUGHT_DELIMITER)/len(ends):.2f} "
          f"eos={ends.count('eos')/len(ends):.2f} cap={ends.count('length')/len(ends):.2f} "
          f"| empty thoughts={sum(x <= 2 for x in lens)/len(lens):.2f} "
          f"| thought tok med={st.median(lens)} p90={sorted(lens)[int(.9*len(lens))]} "
          f"| answered={sum(s['done']=='answer' for s in states)}/{len(states)}", flush=True)


for mode in ("in_reply", "per_turn"):
    states = [{"msgs": [{"role": "user", "content": P.THOUGHT_TEMPLATE.format(question=p["problem"])}],
               "ids": None, "n": 0, "done": None, "lens": [], "ends": [], "gen": 0} for p in probs]
    for s in states:
        s["ids"] = chat(s["msgs"])
    for _ in range(MAX_THOUGHTS):
        act = [s for s in states if s["done"] is None and s["gen"] + CAP <= TOTAL]
        if not act:
            break
        # in_reply: stop at the delimiter and keep generating in the same reply.
        # per_turn: the model ends its turn (EOS) or writes the delimiter; either ends the thought.
        sp = SamplingParams(temperature=0.5, top_p=0.9, top_k=50, max_tokens=CAP,
                            stop=[P.THOUGHT_DELIMITER], include_stop_str_in_output=True,
                            skip_special_tokens=True)
        outs = eng.generate([s["ids"] for s in act], [sp] * len(act))
        for s, o in zip(act, outs):
            s["n"] += 1
            s["gen"] += len(o["ids"])
            s["lens"].append(len(o["ids"]))
            end = (o["stop_reason"] if o["stop_reason"] is not None
                   else ("eos" if o["finish_reason"] == "stop" else o["finish_reason"]))
            s["ends"].append(end)
            if "\\boxed{" in o["text"]:
                s["done"] = "answer"
                continue
            if mode == "in_reply":
                s["ids"] = s["ids"] + o["ids"]
                if end == "eos":
                    s["done"] = "eos"
            else:
                s["msgs"] += [{"role": "assistant", "content": o["text"]},
                              {"role": "user", "content": NEXT}]
                s["ids"] = chat(s["msgs"])
    report(mode, states)
