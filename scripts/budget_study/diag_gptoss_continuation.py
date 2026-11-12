"""Diagnose '!!!!' (token 0) degeneration on gpt-oss continuation calls in vLLM.

Runs thought 1 from a fresh prompt, then thought 2 by continuing from prompt + thought-1
token ids, under different engine settings, one setting per invocation:
    python diag_gptoss_continuation.py <setting>   (setting: cache | nocache | nocache_eager)
"""
import sys

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from thought_ics.budget_study import protocol as P
from thought_ics.datasets import load_dataset_by_name

setting = sys.argv[1]
tp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
kw = {"cache": dict(enable_prefix_caching=True),
      "nocache": dict(enable_prefix_caching=False),
      "nocache_eager": dict(enable_prefix_caching=False, enforce_eager=True)}[setting]
llm = LLM(model="openai/gpt-oss-20b", tensor_parallel_size=tp, max_model_len=32768,
          gpu_memory_utilization=0.85, seed=42, **kw)
tok = llm.get_tokenizer()
probs = load_dataset_by_name("aime", n_problems=5, seed=42)
prompts = [tok.encode(P.THOUGHT_TEMPLATE.format(question=p["problem"]), add_special_tokens=True)
           for p in probs]
sp = SamplingParams(temperature=0.5, top_p=0.9, top_k=50, max_tokens=512,
                    stop=[P.THOUGHT_DELIMITER], include_stop_str_in_output=True,
                    skip_special_tokens=False)
for step in range(6):
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], [sp] * len(prompts),
                        use_tqdm=False)
    for k, o in enumerate(outs):
        ids = list(o.outputs[0].token_ids)
        frac0 = sum(i == 0 for i in ids) / max(1, len(ids))
        print(f"RESULT {setting} tp={tp} step={step} prob={k} ntok={len(ids)} finish={o.outputs[0].finish_reason} "
              f"frac_tok0={frac0:.2f} text={o.outputs[0].text[:80]!r}", flush=True)
        prompts[k] = prompts[k] + ids
