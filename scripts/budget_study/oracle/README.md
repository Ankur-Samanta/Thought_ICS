# Oracle localization: clean vs erroneous prefixes

This tests the paper's localization claims on the final (shared-init) runs. Both methods
localize in the **same first attempt**: the Token-ICS solution is the joined Thought-ICS chain.
So one oracle label per wrong first attempt ("the first erroneous step is k") scores both
methods.

## Steps

```bash
scripts/budget_study/oracle/pipeline.sh build        # -> experiments_budget_study/oracle/batch_<k>.json + manifest.json
# run any number of judges (below), each writes labels_<judge>_<k>.json
scripts/budget_study/oracle/pipeline.sh status       # which judges/batches exist and are valid
scripts/budget_study/oracle/pipeline.sh analyze      # validate, then consensus + clean/erroneous table
```

The build step:
- samples about 30 wrong first attempts per model from the math datasets (AMC23 6, AIME 8,
  MATH-500 8, MathQA 8),
- skips grading false negatives (answers that differ from the ground truth only in formatting),
- splits the sample into 3 batches,
- is deterministic: the same results give the same batches.

## Adding a judge (any model, any tool)

A judge reads `batch_<k>.json` and writes `labels_<judge>_<k>.json`. The judge name must not
contain `_`. The full instructions and output format are in `judge_prompt.md`. To print the
complete task text for one judge and batch:

```bash
scripts/budget_study/oracle/pipeline.sh task <judge> <k>
```

- **Codex:** `scripts/budget_study/oracle/run_codex.sh <model>`. It runs every batch, writes
  `labels_codex-<model>_<k>.json`, skips batches already done, and validates the result.
- **Claude Code subagents** (Opus / Sonnet / Fable): give each subagent the output of
  `pipeline.sh task <name> <k>`.
- **Anything else:** paste the task text into the tool, and save its JSON answer to the
  output path it names.

`validate.py` checks every label file: every id appears exactly once, `first_error_step` is an
int in `[0, n_steps]`, and `confidence` is one of `high`, `medium`, `low`. `analyze` only runs
if every label file passes.

## Analysis

- **Consensus:** a step chosen by at least 2 judges, otherwise the median of the judges' steps.
  Agreement is reported as unanimous and within ±1 step. Items whose consensus is 0 (no error
  found) are dropped.
- **Classifying a localization:** each method's first localization is
  - **exact** or **earlier** (both are a clean prefix),
  - **later** (an erroneous prefix), or
  - **failed** (an unusable localization).
- **Mapping Token-ICS:** a Token-ICS cut maps to the step that contains it.
- **Outcome:** the fix rate of the resample after a clean prefix vs after an erroneous one, per
  model and method.
