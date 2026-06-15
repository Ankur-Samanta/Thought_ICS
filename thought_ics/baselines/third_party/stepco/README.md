# StepCo baseline

Faithful re-implementation of **StepCo** (Wu et al., 2024) — *Enhancing
Mathematical Reasoning in LLMs by Stepwise Correction*
([arxiv:2410.12934](https://arxiv.org/abs/2410.12934),
[github:wzy6642/StepCo](https://github.com/wzy6642/StepCo)) — adapted to our
3p_baselines harness.

## What's faithful to the original

- **Algorithm:** verbatim port of `wzy6642/StepCo/StepCo/solving_pipeline.py`
  (`initialization` → `pipeline` → `rectification`). Same OSV/PSV checks,
  same termination conditions (OSV passes, two consecutive identical answers,
  or max iterations).
- **Prompts:** verbatim copies of `zero_shot_cot_prompt_template`,
  `get_numerical_answer_prompt_template`, and
  `stepwise_rectify_prompt_template_v2` from their `prompt_template.py`.
- **Step parsing:** verbatim `get_reasoning_steps` and
  `find_first_smaller_index` from their `utils.py`.
- **Verifier:** **Math-Shepherd** (`peiyi9979/math-shepherd-mistral-7b-prm`),
  which is the default `verification_model` in StepCo's released `config.py`.
- **Hyperparameters:** `threshold=0.5`, `max_iterations=5` (StepCo defaults).

## What's different (and why)

- **Generation backend.** StepCo's reference code calls GPT-4o via
  `answered_by_openai`. We replace this with a vLLM HTTP client to a locally
  hosted reasoner so we can run the same 8-model spread used elsewhere in
  the paper. The substitution is purely at the *generation transport* layer;
  prompts, parameters, and the algorithm are unchanged.
- **Verifier backend.** StepCo's `verification.py` loads Math-Shepherd
  in-process via HF transformers and reads raw logits. We instead query a
  vLLM HTTP server using vLLM's `prompt_logprobs` extension. The protocol
  appends a `+` placeholder token after each `ки` so that the
  `prompt_logprobs` entry at the placeholder position contains the model's
  predicted distribution over `{+, -}` at the position right after `ки`.
  This is mathematically equivalent to StepCo's
  `model(input_ids).logits[:, :, [+, -]]` indexed at `input_ids == ки`.
- **Generation hyperparameters.** Defaults match the rest of the
  3p_baselines harness (`temperature=0.5`, `max_tokens=2048`, `top_p=0.9`,
  `top_k=50`) rather than StepCo's own defaults (0.7 / 2048 / 0.95 / 0).
  This keeps StepCo on the same eval footing as Self-Refine and CoVe in our
  comparison. Override with `--generation-temp`, `--max-tokens`, etc.
- **Answer normalization.** StepCo's `post_process_value` calls `eval()` on
  numeric strings, which doesn't work for our multiple-choice benchmarks
  (CSQA, GPQA). We use the harness-wide `normalize_answer` from
  `dataset_loaders.py` instead, after extracting `\\boxed{}` from the
  answer-extraction LLM call.

## GPU layout

3 GPUs total:

| GPUs | Model                                          | Port | TP |
|------|------------------------------------------------|------|----|
| 0, 1 | `peiyi9979/math-shepherd-mistral-7b-prm`       | 8002 | 2  |
| 2    | base reasoner (e.g. Llama-3.1-8B-Instruct)     | 8001 | 1  |

## How to run

**1. Start the two vLLM servers** (in a separate terminal or as background jobs):

```bash
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  ./3p_baselines/stepco/launch_servers.sh
```

Wait until both server logs show `Uvicorn running on http://...`.

**2. Run the eval:**

```bash
python 3p_baselines/stepco/eval_stepco.py \
  --base-model-url http://localhost:8001 \
  --base-model-name meta-llama/Llama-3.1-8B-Instruct \
  --verifier-url http://localhost:8002 \
  --model llama8b \
  --dataset math500 \
  --n-problems 100
```

Results land in `experiments/Baseline_StepCo_<model>_<dataset>_<timestamp>/`
with the same `config.json` / `results.json` / `metrics.json` schema as the
other 3p baselines.

## Caveats

- **Math-Shepherd is math-trained.** It was supervised on GSM8K + MATH step
  labels. Scores on CSQA and GPQA reasoning are out-of-distribution and
  should be interpreted as such — this is a deliberate property of StepCo
  with this verifier (not a bug in our port). Document this in any
  comparison table.
- **`<Step>` tag adherence.** StepCo's CoT prompt asks the model to wrap
  each step in `<Step N> ... </Step N>`. Different base reasoners follow
  this instruction with different fidelity. The step parser
  (`get_reasoning_steps`) is robust to missing/extra tags but degrades to
  one giant "step" if the model produces no tags at all — the verifier will
  then return a single global score, and rectification becomes a no-op
  unless that score is below threshold.
