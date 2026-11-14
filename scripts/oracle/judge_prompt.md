# Oracle error localization: instructions for a judge

You are an expert grader. Your job is to locate the **first erroneous step** in step-by-step
solutions that reached a **wrong final answer**. Each solution comes with the correct answer.

## Input

Read the batch file given to you, e.g. `results/oracle/part1/batch_0.json`. It is a
JSON list. Each item has these fields:
- `id`
- `problem`
- `ground_truth`: the correct final answer
- `final_answer_given`: the solution's wrong answer
- `steps`: the solution's steps, in order. Step 1 is `steps[0]`.

## Task

For each item, find the **first step that contains an error the wrong answer depends on**. Count
as errors: arithmetic or algebra mistakes, wrong formulas or facts, misreading the problem,
invalid or unjustified deductions, wrong case analysis, and dropped constraints.

- Judge each step on its own correctness, given the steps before it. A correct step that
  merely leads nowhere is **not** an error.
- If a step is correct but incomplete, and a later step draws the wrong conclusion, the later
  step is the error.
- Work the problem yourself as needed to verify the steps. Use the correct answer as a guide,
  not as proof.
- If you cannot find any erroneous step, use `0`. For example, the solution may actually be
  correct and only formatted differently from the ground truth.

## Output

Write a JSON list to the output file you were given, e.g.
`results/oracle/part1/labels_<judge>_0.json`. Write one object per item, in any order:

```json
{"id": "<item id>", "first_error_step": <int: 1-indexed step, or 0>, "confidence": "high|medium|low", "note": "<= 25 words: what the error is"}
```

Write only that file. Do not modify the batch files. Every item in the batch needs exactly one
object.
