"""The six evaluation benchmarks, frozen to the exact problems and order used in the paper.

``data/benchmarks/<name>.json`` holds each problem set: AMC23 (40 problems), and 100 problems each
from AIME (1983-2024), MATH-500 (level 5), CommonsenseQA and MathQA. GPQA's terms ask that its
questions not be published in plain text, so for GPQA only the problem ids and SHA-256 hashes are
stored (``gpqa_ids.json``); the problems are rebuilt from the Hugging Face dataset
(``Wanfq/gpqa``, ``gpqa_main``) and checked against the hashes.

Each problem is ``{"problem", "answer", "unique_id"}``. Multiple-choice problems (CSQA, GPQA,
MathQA) include the options and answer-format instruction in the problem text; their answer is a
choice letter.
"""
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"
BENCHMARKS = ["amc23", "aime", "math500", "csqa", "gpqa", "mathqa"]
MATH_BENCHMARKS = {"amc23", "aime", "math500"}   # graded by mathematical equivalence (Math-Verify)


def run_tag(name: str) -> str:
    """Benchmark name as used in run-directory names (MATH-500 is its level-5 subset)."""
    return "math500_L5" if name == "math500" else name


def load_benchmark(name: str) -> List[Dict[str, str]]:
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")
    if name == "gpqa":
        return _load_gpqa()
    rows = json.load(open(DATA_DIR / f"{name}.json"))
    return [{"problem": r["problem"], "answer": r["answer"], "unique_id": r["id"]} for r in rows]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _load_gpqa() -> List[Dict[str, str]]:
    """Rebuild the GPQA problems (options shuffled with seed 42, as when the runs were made) and
    check every question and answer against the stored hashes."""
    from datasets import load_dataset
    ds = load_dataset("Wanfq/gpqa", name="gpqa_main", split="train")
    rng = random.Random(42)
    built = {}
    for idx, item in enumerate(ds):
        choices = [item["Correct Answer"], item["Incorrect Answer 1"], item["Incorrect Answer 2"],
                   item["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        problem = (f"{item['Question']}\n A) {shuffled[0]}\n B) {shuffled[1]}\n C) {shuffled[2]}\n D) {shuffled[3]}\n\n"
                   "Your final answer should be a single choice letter in the form \\boxed{answer}, at the end of your response.")
        built[f"gpqa_{idx}"] = {"problem": problem, "answer": "ABCD"[order.index(0)], "unique_id": f"gpqa_{idx}"}
    out = []
    for ref in json.load(open(DATA_DIR / "gpqa_ids.json")):
        p = built[ref["id"]]
        if _sha(p["problem"]) != ref["sha256"] or _sha(p["answer"]) != ref["answer_sha256"]:
            raise RuntimeError(f"GPQA problem {ref['id']} does not match the evaluated version; "
                               "the Hugging Face dataset may have changed.")
        out.append(p)
    return out


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (case-insensitive, whitespace stripped).

    This is particularly important for multiple choice questions where
    'a' should match 'A', ' B ' should match 'B', etc.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer (lowercase, stripped, extra spaces removed)
    """
    if not answer:
        return ""

    # Convert to lowercase
    answer = answer.lower()

    # Strip leading/trailing whitespace
    answer = answer.strip()

    # Replace multiple spaces with single space
    answer = re.sub(r'\s+', ' ', answer)

    return answer
