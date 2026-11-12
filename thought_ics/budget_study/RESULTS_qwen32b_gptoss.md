# Results: qwen32b, gpt-oss-20b, gpt-oss-120b (final protocol, 2026-11-12)

The final protocol (see README):
- Shared first attempts: Token-ICS starts from the joined Thought-ICS chain, under the same prompt.
- A failed localization uses up the attempt.
- Uniform budgets: 16k generation and 8k localizer.
- Thoughts are paragraph steps generated at inference time.
- L2 (binary oracle), 10 attempts.
- Math datasets are regraded with math_verify (exact truncation at the first correct attempt); strict grading is in brackets.
- Runs: `experiments_budget_study/full/*_shared_fbattempt_*`.
- Report: `scripts/budget_study/report.py`.

## Accuracy (540 problems per model, 6 datasets)

| model | first attempt | Thought-ICS final (lift) | Token-ICS final (lift) | Thought − Token [95% paired CI] |
|---|---|---|---|---|
| qwen32b | 60.9 [58.9] | 75.6 (+14.6) | 74.6 (+13.7) | +0.9 [−1.1, +3.0] |
| gpt-oss-20b | 58.3 [55.6] | 79.8 (+21.5) | 74.8 (+16.5) | **+5.0 [+2.0, +8.0]** |
| gpt-oss-120b | 64.8 [60.7] | 83.3 (+18.5) | 80.0 (+15.2) | **+3.3 [+0.9, +5.7]** |

The largest gaps are on the hardest datasets:

| model | AIME (Thought / Token) | MATH-500 (Thought / Token) |
|---|---|---|
| gpt-oss-20b | 51 / 37 | 85 / 77 |
| gpt-oss-120b | 59 / 50 | 91 / 86 |

## Localization mechanics (all attempts, after regrade)

| model | failed localizations (Thought / Token) | fix rate, resample from the localized prefix (Thought / Token) |
|---|---|---|
| qwen32b | 0% / 17% | 4.7% / 5.1% |
| gpt-oss-20b | 0% / 43% | 7.4% / 9.0% |
| gpt-oss-120b | 0% / 21% | 7.4% / 7.3% |

## Oracle localization

- Scope: all 616 wrong first attempts, with grader false negatives removed.
- Judges: Opus, Sonnet and Haiku, labeling independently.
- Pairwise exact agreement: 73–81%; within ±1 step: 84–87%.
- Consensus: a step chosen by at least 2 judges, otherwise the median.
- Pipeline: `scripts/budget_study/oracle/`. Labels: `experiments_budget_study/oracle_full/`.

| model | Thought-ICS clean prefix | Token-ICS clean prefix | difference [95% paired CI] |
|---|---|---|---|
| qwen32b (n=207) | 71% | 54% | +16.9 [+10.1, +24.2] |
| gpt-oss-20b (n=213) | 83% | 58% | +24.9 [+16.9, +32.9] |
| gpt-oss-120b (n=180) | 90% | 67% | +22.8 [+16.1, +29.4] |
| pooled (n=600) | | | **+21.5 [+17.3, +25.8]** |

Clean = localized at or before the consensus first error. A failed Token-ICS localization counts as not clean.

| method | fixed after a clean prefix | fixed after an erroneous prefix | ratio | difference [95% CI] |
|---|---|---|---|---|
| Thought-ICS | 16.5% (n=485) | 7.0% (n=115) | 2.4× | **+9.5 [+3.8, +15.0]** |
| Token-ICS | 12.4% (n=356) | 8.4% (n=119) | 1.5× | +4.0 [−2.1, +9.9] |

## Data-quality findings that affect the paper

- **The paper's string grader is wrong on many MATH-500 answers.** It marks 22–44% of these models' "wrong" MATH-500 first attempts wrong when they are actually correct: `5.5` vs `\frac{11}{2}`, `x=5` vs `5`, `120^\circ` vs `120`.
- **MathQA and GPQA contain label noise.** The judges flagged several items whose reference answer is wrong or whose options are inconsistent.
- **MathQA's letter grading misses some correct answers.** Some correct answers are written as the option's value rather than its letter.
