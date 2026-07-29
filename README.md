# Thought-ICS: Structure Enables Effective Self-Localization of Errors in LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2602.02416-b31b1b.svg)](https://arxiv.org/abs/2602.02416)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📄 **Paper:** [arXiv:2602.02416](https://arxiv.org/abs/2602.02416)

**Iterative Correction Sampling of Thoughts (Thought-ICS)** — a framework for studying
self-correction in language models by generating reasoning *thought-by-thought*, localizing
the first erroneous step, and backtracking to resample from the last correct point.

> **TL;DR.** We show that language models can explicitly self-localize errors in incorrect reasoning
> when it is structured as discrete, semantically coherent thought steps, enabling effective
> self-correction through targeted backtracking and resampling.

This repository contains the method, baselines, and evaluation harness.

---

## The idea

Self-correction for LLMs is usually done by **critiquing an answer and regenerating the entire
reasoning trace**. We show that **targeted backtrack-and-resample works better**: self-localize the
first erroneous step, then sample a counterfactual from there — self-correction as a *targeted
intervention* rather than wholesale regeneration.

The crux is **self-localization**. Rather than asking whether LLMs can localize errors in arbitrary
traces, we take a different angle: **generate the reasoning so that localization is tractable in the
first place** — as a discrete chain of semantically coherent thoughts with clear boundaries.

Inspired by how the brain's anterior cingulate cortex monitors errors at discrete decision points
(not individual neurons, not whole behaviors), Thought-ICS generates reasoning as discrete thoughts,
localizes the first error at that granularity, and resamples from the last correct step.

### Thought MDP (thought-by-thought generation)

Standard CoT is a single token-level rollout. A **Thought MDP** keeps the same model but changes
the *action space*: each action is a complete reasoning step (a "thought"), delimited by a stop
marker the model emits. Same `p_θ`, different prompting — but now the trace has principled
boundaries that can be traversed, verified, and edited step by step.

```
Algorithm 1 — Thought-by-thought generation
  Input: question x, policy π_thought, max depth D
  s ← x
  repeat
      a ~ π_thought(· | s)      # sample next thought
      s ← s ⊕ a                 # append thought to state
  until s contains an answer or max depth reached
  return s
```

### The Thought-ICS correction loop

<p align="center"><img src="figures/tree_visualization.png" width="95%"></p>

**Generation → Verification → Localization → Resampling**, iterated. Verification gates the loop
(self- or oracle-verification). On a flagged response, the model is shown its full numbered trace
and asked for the *first* erroneous thought; the scaffold backtracks to that point and resamples a
new continuation from the shared prefix. The loop exits on one of three conditions:
**(1) Verified Accuracy**, **(2) V/L Disagreement** (verifier flags an error, localizer finds none),
or **(3) MaxIter**.

```
Framework 1 — Thought-ICS
  Input: question x, max iterations L
  τ ← Generate(x)                          # Thought MDP (Alg. 1)
  repeat
      return τ if Verify(τ)                # (1) Verified Accuracy
      e ← localize first erroneous thought in τ
      return τ if e = ∅                    # (2) V/L Disagreement
      τ_prefix ← τ[:e]                     # backtrack to thought e−1
      τ ← Generate(τ_prefix)               # resample correction
  until max iterations                     # (3) MaxIter
  return τ
```

**Token-ICS** is the unstructured baseline: the same localize → backtrack → resample loop, but on a
continuous CoT trace where the model must *quote* the erroneous span instead of naming a step.

---

## Results

We evaluate **8 models (3B–120B)** — LLaMA-3 (3B/8B/70B), Qwen-2.5 (7B/14B/32B), GPT-OSS (20B/120B) —
across **6 benchmarks**: AMC23, AIME, MATH500-L5, MathQA, CSQA, GPQA. Figures are averaged over all six
datasets unless noted.

### 1. Structure improves self-correction under oracle verification

Using an oracle to filter for genuinely-wrong solutions, both methods backtrack-and-resample from a
localized error. Self-correction lift is **larger and more uniform for Thought-ICS (red)** than for
unstructured Token-ICS (green). Part of the lift comes from thought-by-thought generation itself
(light red); the rest from localization-driven correction (solid red).

<p align="center"><img src="figures/l2_errorbar.png" width="80%"></p>

### 2. Structure enables precise self-localization

The objective when backtracking is a **clean prefix**: a sequence of thoughts free of any decision
that would derail the reasoning (self ≤ oracle); otherwise the prefix is *erroneous*. Within
structured reasoning, models reach a clean prefix far more often — and the gap grows with scale.

<p align="center">
<img src="figures/self_localization_vs_oracle.png" width="46%">
<img src="figures/edit_accuracy_clean_vs_erroneous.png" width="46%">
</p>

*Left:* clean prefixes (exact match + earlier) vs erroneous, per model.
*Right:* this verifies the premise of backtrack-and-resample directly — sampling counterfactuals from
a **clean prefix corrects 2–4× more often** than from an erroneous one. Precise localization is what
makes the correction land.

The deviation between self- and oracle-localization is **tight and centered at zero for large
models**, confirming the lift comes from precise localization, not conservatism (just blaming step 1):

<p align="center"><img src="figures/localization_error_distribution.png" width="65%"></p>

> Oracle localization is a consensus of three frontier models (Claude-Sonnet-3.7, GPT-4.1,
> GPT-5-mini): 51% unanimous, 74% within ±1 step, 85% within ±2 steps.

**Without structure, localization is inconsistent.** On unstructured CoT, models localize *later*
than the oracle and miss root causes — only **30–45% clean prefixes vs 60–80% for Thought-ICS**:

<p align="center"><img src="figures/self_localization_vs_oracle_token.png" width="65%"></p>

### 3. A fully autonomous system (no oracle)

> **Autonomy levels.** The `--autonomy-level` flag controls how much oracle support the
> verify→localize→resample loop receives — each level strips away one more piece of external help:
> - **L1 — Oracle:** ground truth drives both verification and localization (the model is told *where* the error is).
> - **L2 — Binary oracle:** the model is told only *that* the trace is wrong; it must self-localize the erroneous step.
> - **L3 — Autonomous:** no oracle — the model self-verifies *and* self-localizes. This is the fully
>   autonomous setting these results target (and what the Thought-ICS-A confidence safeguard protects).
>
> **Conditioning the resample (`--context`, orthogonal to the level above).** By default the counterfactual
> is a fresh **stochastic sample** from the clean prefix. Passing `--context` instead *conditions* that resample
> on the previous failed chain and the error analysis from the prior iteration. This conditioning is
> independent of the oracle level and can be combined with any of L1/L2/L3. We find that small LMs cannot
> yet make effective use of this explicit error feedback, so **the paper focuses on stochastic sampling from
> the clean prefix**; conditioning is provided for completeness and as a direction for larger models.

Zooming in on how the sub-skills of self-correction **scale**: localization improves with model size
(bigger models are genuinely more precise), and resampling tracks localization quality. But
**self-verification does not reliably scale** — making it the bottleneck for autonomous
self-correction.

**Self-verification degrades over iterations.** Stratified by terminal iteration, *recall* rises but
*specificity* collapses — models increasingly fail to recognize correct answers and break them:

<p align="center"><img src="figures/self_verification_by_terminal_iteration_part1a.png" width="65%"></p>

Averaged over iterations, every self-verification gating strategy **breaks more than it fixes**.
Sampling 9 verifications per step and varying strictness (Any / Majority / Unanimous) only shifts the
recall–specificity trade-off:

| Method     | Recall | Specificity | Broke | Fixed |
|------------|:------:|:-----------:|:-----:|:-----:|
| Single     | 68.3%  | 66.9%       | 10.9% | 6.2%  |
| Any        | 94.6%  | 30.9%       | 21.4% | 8.1%  |
| Majority   | 70.2%  | 69.9%       | 12.7% | 6.3%  |
| Unanimous  | 34.7%  | 94.6%       | 6.0%  | 3.5%  |

**Confidence safeguard → Thought-ICS-A.** Breakage concentrates in two exit conditions. Resetting to
the initial response on those low-confidence exits (and only keeping corrections that pass verified
accuracy) turns the method net-positive:

| Termination condition       | Broke | Fixed | Net Lift |
|-----------------------------|:-----:|:-----:|:--------:|
| (1) Verified Accuracy       | 2.5%  | 6.4%  | **+3.9%** |
| (2) V/L Disagreement        | 51.3% | 2.0%  | −49.3%   |
| (3) MaxIter                 | 77.4% | 5.4%  | −72.0%   |

<p align="center"><img src="figures/self_verification_by_terminal_iteration_part1b.png" width="65%"></p>

**Toward fully autonomous self-correction.** Run with no oracle (Thought-ICS-A, with the confidence
safeguard), this shows the benefits of backtracking and resampling within structured reasoning over
unstructured-CoT full-regeneration baselines such as **Self-Refine** and **CoVe**. The benefits
surface mainly in **larger models (~14B and up)**, where self-localization is precise enough for
targeted correction to pay off; smaller models, with noisier localization and weaker
self-verification, see limited gains.

<p align="center"><img src="figures/self_correction_lift_errorbar.png" width="65%"></p>

> Additional analyses (token-level localization, per-model self-verification, chain-length
> distributions, compute efficiency) and the figures behind them are in [`figures/`](figures/).

**Takeaway.** By structuring reasoning as a chain of semantically coherent thoughts *at generation
time*, self-localization becomes tractable — enabling self-correction as a targeted intervention
rather than wholesale regeneration.

---

## Repository structure

```
thought_ics/                     # the package
├── thought_mdp.py               # thought-by-thought generation (Agent / Environment / TreeSearch)
├── self_correction.py           # the Thought-ICS correction loop (localize → backtrack → resample)
├── chain_cache.py               # caches initial chains across runs
├── datasets.py                  # loaders for AMC23/AIME/MATH500/MathQA/CSQA/GPQA
├── metrics.py                   # self-correction metrics
├── recommended_prompts.py       # ⭐ refined prompts + delimiter + knobs — the ACTIVE defaults
├── paper_prompts.py             # original paper prompts, kept for reference / exact reproduction
├── models/                      # vLLM model-management layer (BaseModelManager, config)
├── baselines/
│   ├── token_ics.py             # Token-ICS (unstructured-CoT baseline)
│   ├── cot_eval.py              # plain / iterative CoT baselines
│   └── third_party/             # Self-Refine, Chain-of-Verification, StepCo re-implementations
├── localization/                # oracle error-localization pipeline (export → judge → match → analyze)
├── verification/                # majority-vote self-verification experiments
└── eval/
    └── batch_eval.py            # main evaluation entry point

data/        benchmark datasets + few-shot prompts
scripts/     helper / exploration scripts
figures/     paper figures (rendered in this README)
cache/       cached initial reasoning chains
```

---

## Installation

Requires Python ≥ 3.9 and a CUDA GPU for local inference (vLLM). API-only runs need no GPU.

Dependencies are managed with **pip via `requirements.txt`**. If you already have a suitable Python:

```bash
pip install -r requirements.txt   # all library dependencies
pip install -e .                  # the thought_ics package itself
```

`environment.yml` is provided only to provision the Python version and an isolated conda env —
libraries still install through pip:

```bash
conda env create -f environment.yml   # creates the 'thought-ics' env (Python 3.12 + pip)
conda activate thought-ics
pip install -r requirements.txt
pip install -e .
```

---

## Usage

All entry points are run as modules (`python -m thought_ics.<...>`).

**Thought-ICS (thought-by-thought generation + correction).** Set `--autonomy-level` to
`1` Oracle, `2` Binary oracle, or `3` Autonomous; add `--context` to condition the resample on the
prior iteration's error (any level). See [**Autonomy levels**](#3-a-fully-autonomous-system-no-oracle)
above for what each means.

```bash
# Autonomous self-correction (L3) with a local model
python -m thought_ics.eval.batch_eval \
    --autonomy-level 3 --dataset math500 --model llama8b \
    --gpus 0,1 --tensor-parallel-size 2 --n-problems 100

# Oracle-verification setting (L2): model is told it's wrong and must self-localize
python -m thought_ics.eval.batch_eval \
    --autonomy-level 2 --dataset aime --model qwen7b \
    --gpus 0 --tensor-parallel-size 1
```

Available `--model`: `llama3b llama8b llama70b qwen7b qwen14b qwen32b gptoss20b gptoss120b`.
Available `--dataset`: `amc23 aime math500 mathqa csqa gpqa`.

**Run via an OpenAI-compatible API** instead of a local GPU (`--3p` routes all inference through the API):

```bash
# No GPU needed: --gpus / --tensor-parallel-size are not required under --3p
python -m thought_ics.eval.batch_eval \
    --autonomy-level 3 --dataset amc23 \
    --3p --3p-model gpt-4o --3p-api-key $OPENAI_API_KEY
```

For an API-only installation (including Windows), omit the local inference stack:

```bash
pip install -r requirements-api.txt
pip install -e . --no-deps
```

The API configuration can be supplied with environment variables. `OPENAI_BASE_URL` is optional
for the official OpenAI endpoint and enables any OpenAI-compatible provider; `OPENAI_MODEL` defaults
to `gpt-4o` when omitted.

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"  # optional
$env:OPENAI_MODEL = "your-model-name"                      # optional
.\scripts\run_api.cmd -Dataset amc23 -NProblems 1
```

The equivalent CLI options are `--3p-api-key`, `--3p-base-url`, and `--3p-model`. Prefer the
environment variable for the API key so it does not appear in shell history or process arguments.
The `.cmd` launcher works even when direct PowerShell script execution is disabled and does not
change the machine's persistent execution policy.

**Baselines:**

```bash
# Token-ICS: unstructured-CoT correction where the model quotes the erroneous span
# (--mode selects the built-in problem set: simple, amc, or both; --gpu is a single GPU id)
python -m thought_ics.baselines.token_ics --mode both --model llama8b --gpu 0 --n-problems 100

# Iterative-CoT baselines (--baseline-type: single | iterative_l1..l3 | iterative_no_gt |
# iterative_with_gt | majority_vote; add --context for historical-context conditioning)
python -m thought_ics.baselines.cot_eval \
    --baseline-type iterative_l2 --dataset math500 --model llama8b \
    --gpus 0 --tensor-parallel-size 1

# Self-Refine and Chain-of-Verification (--method: self_refine | cove)
python -m thought_ics.baselines.third_party.eval_3p_baselines \
    --method self_refine --dataset math500 --model llama8b --gpus 0
```

**Oracle error-localization pipeline** (used to build the localization figures):

```bash
# 1) export localization prompts from completed runs
python -m thought_ics.localization.export_prompts --experiments "experiments/eval_*" --output prompts.jsonl
# 2) judge them offline with frontier models -> responses.jsonl
# 3) match judge responses back and compute agreement
python -m thought_ics.localization.match_results --prompts prompts.jsonl --responses responses.jsonl
```

**Metrics & majority-vote verification:**

```bash
python -m thought_ics.metrics experiments/<run_dir>                 # compute metrics.json for a run
python -m thought_ics.verification.mv_verification --model llama3b --dataset aime --n-problems 100
```

Each run writes a directory under `experiments/` containing `results.json` (full traces),
`metrics.json`, and `config.json`.

> **Note:** this is a code release. The raw experiment outputs and judge responses that back the
> paper figures are not shipped; the commands above regenerate them.

### Prompts: refined defaults (recommended) vs. paper originals

By default the pipeline now loads the **refined** thought-by-thought generation and
error-localization prompts, delimiter, and knobs from
[`thought_ics/recommended_prompts.py`](thought_ics/recommended_prompts.py) — these come from our
follow-up experiments and give the best results. They sharpen *originating-cause* localization
(target the step where reasoning first derailed, not where the wrong answer surfaces), use cleaner
step-by-step generation, stop only on `</thought>`, and set `max_thoughts=20`,
`max_tokens_per_thought=512`, `max_ics_iterations=10` (temperatures unchanged).

The **exact prompts used in the paper** are preserved in
[`thought_ics/paper_prompts.py`](thought_ics/paper_prompts.py) for reference / exact reproduction
(pass `paper_prompts.GENERATION_PROMPT` to `ToTEnvironment(prompt_template=...)` and use
`paper_prompts.localization_prompt(...)`). Note: fully-autonomous L3 localization keeps its
original prompt (it must be able to report "no error" via `\boxed{0}`).

---

## Citation

```bibtex
@misc{samanta2026structureenableseffectiveselflocalization,
      title={Structure Enables Effective Self-Localization of Errors in LLMs},
      author={Ankur Samanta and Akshayaa Magesh and Ayush Jain and Kavosh Asadi and Youliang Yu and Daniel Jiang and Boris Vidolov and Kaveh Hassani and Paul Sajda and Jalaj Bhandari and Yonathan Efroni},
      year={2026},
      eprint={2602.02416},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.02416},
}
```

## License

Released under the [MIT License](LICENSE).
