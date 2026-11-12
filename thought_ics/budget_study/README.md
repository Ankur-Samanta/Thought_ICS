# Budget-controlled Thought-ICS vs Token-ICS study

This study reruns the oracle-verified (L2) comparison for **qwen32b, gpt-oss-20b and gpt-oss-120b**
with every budget matched and non-binding. In the paper runs:

- Token-ICS CoT was capped at 2048 tokens. 20–43% of its solutions for these models were cut off
  before an answer and were scored wrong.
- The localizer budgets differed: 512 tokens for Thought-ICS (the unset default) and 1024 for Token-ICS.

Both methods run in the same job on the same problems and the same vLLM engine
(`run_study.py --method both`).

## Protocol (identical across models and methods unless stated)

| knob | value |
|---|---|
| context window | 32,768 (qwen32b's limit; uniform for scaling) |
| generation budget per solution/chain (incl. kept prefix on resample) | 16,384 |
| localizer budget | 8,192 |
| per-thought cap (safety only) | 2,048, and never more than the remaining budget |
| max thoughts | 500 (the token budget binds first) |
| sampling | temperature 0.5, top-p 0.9, top-k 50; localizer temperature 0.5 |
| ICS | L2 (binary oracle), 10 iterations, stochastic resample from the kept prefix |
| grading | last `\boxed{}` over the full text, `normalize_answer` equality |
| format | chat template for every call; gpt-oss: empty analysis block, then the final channel |
| empty output | regenerate, up to 3 retries |

**Thought-ICS** (inference-time structure, no retroactive splitting):

- Each thought is one paragraph. Every generation call stops at a blank line (`\n\n`).
- The next thought is generated from the exact token ids so far, so the chain is a sample of
  the model's own reasoning.
- Content-free paragraphs (e.g. `---`) are attached to the previous thought.
- The chain runs until the reply ends, the same as Token-ICS.
- Localizer: SRPO's final L2 prompt (`l2new`). The parser checks the last `\boxed{N}`, then the
  last "Step N", then falls back to step 1.
- Resample: keep the thoughts before the flagged step (exact ids) and continue.

**Token-ICS:**

- One CoT reply.
- Localizer: the same SRPO prompt wording, asking for an exact quote `\boxed{ERROR_QUOTE: "..."}`.
- The quote must match the solution exactly. Only whitespace and Unicode differences are ignored;
  a paraphrased or invented quote still fails.
- If the quote isn't found, regenerate from scratch (the paper's behavior).
- Resample: cut the solution before the quote and continue from that prefix.

## Deviations from SRPO / the paper, and why

1. **Chat template for generation.**
   - SRPO encoded raw text. That's fine for OLMo and Qwen, but wrong for gpt-oss: raw mode
     produced harmony control tokens inside the text.
2. **Thought delimiter: blank line, not `</thought>`.**
   - gpt-oss ignores `</thought>` in every prompt and format tested (5–32% of thoughts used it).
   - Qwen ends its chat reply right after `</thought>` (47% empty thoughts, 2/20 answered).
   - Blank lines give multi-step chains for all three models: 19–20/20 answered.
   - See `scripts/budget_study/diag_thought_format.py` and `diag_thought_turns.py`.
3. **Prompt.**
   - SRPO's step criteria are kept verbatim.
   - The instruction now reads "one step per paragraph; no separators; continue until you
     reach the final answer". The last part is the paper prompt's wording, and both methods get it.
   - SRPO's "write one step at a time" wording made Qwen stop after one step.
4. **Budgets.**
   - SRPO's 5120 generation / 4096 localizer / 512 per-thought limits bind for these models,
     so all are raised as in the table, the same for both methods.
   - Thought-ICS may use the full remaining budget. SRPO's rule ("stop if a full-cap thought
     no longer fits") would have given it up to 2,048 fewer tokens.
5. **gpt-oss engine.**
   - Prefix caching is off. With TP>1 plus prefix caching, vLLM 0.11.2 on A100 degenerates to
     token 0 (`!!!!`) on continuation calls (25/30 at TP=2). Without caching: 0/30.
   - See `scripts/budget_study/diag_gptoss_continuation.py`.
   - The paper's gpt-oss runs used TP 2/4 and may have been affected.
6. **Fairness-audit fixes (commit d88926f):**
   - Thought chains continue after a `\boxed{}` until the reply ends (Token-ICS already did).
   - Integer EOS ids are recognized as end of reply.
   - The empty-output retry applies to Thought-ICS too.
   - The Token-ICS parser accepts quote *formats* as leniently as Thought-ICS accepts step formats.

Known, disclosed asymmetries (both make the comparison harder for Thought-ICS, never easier):

- A quote that appears several times cuts at its first occurrence.
- The Token-ICS prefix is re-tokenized after the text cut.

## Running

```bash
scripts/budget_study/submit.sh smoke <model> [--n-problems 20 --max-ics-iterations 1]  # -> experiments_budget_study/dev
scripts/budget_study/submit.sh all all                                                 # 18 jobs -> experiments_budget_study/full
```

Each run directory contains `config.json` (all knobs + git commit), `{thought,token}_results.jsonl`
(every generation, localization prompt/response, parse status, cut, budget status) and
`summary.json`. The summary includes accuracy, budget-hit rates, localizer cap and parse rates,
quote-not-found rate, and generation and localization token totals.

Still to do after the runs: oracle localization (Sonnet-3.7 / GPT-4.1 / GPT-5-mini) on the new
localizations for both methods. It's paid API calls and needs approval.
