"""Recommended prompts (updates since the paper) — use these for best results.

The prompts in this module are *refined* versions of the thought-by-thought generation
and error-localization prompts, developed in follow-up experiments after the original
Thought-ICS paper. In our experience they produce better self-correction: cleaner
step-by-step generation and sharper, originating-cause error localization.

These are now the **default** prompts/delimiter/knobs loaded by the pipeline:
  - `thought_ics.thought_mdp`     (thought-by-thought generation, ``ToTAgent`` / ``ToTEnvironment``)
  - `thought_ics.self_correction` (error localization, ``identify_error_step`` for L1/L2)
The exact prompts used for the published paper experiments are preserved, unchanged, in
`thought_ics.paper_prompts` for reference / exact reproduction.

Why these are better
--------------------
Thought-by-thought generation:
  - An explicit "one reasoning step at a time / do not solve it all at once" framing that
    asks for self-contained, *substantive* steps (no filler, no restating the problem) that
    cohere logically with prior steps — yielding cleaner, more separable thought boundaries.

Error localization (binary / no ground truth — the recommended default):
  - Targets the **originating** error (the step where a decision or assumption first derailed
    the reasoning), not the later step where the wrong answer becomes visible. Errors compound,
    so a subtle early mistake that "looks fine" for several steps is still the correct target.
  - Operational test: "a step is erroneous if you cannot justify its claims from the problem
    statement and earlier verified steps alone; stop at the first step you cannot justify."
  - Forces a single, mandatory ``\\boxed{N}`` final line (the step index, *not* the answer).

Conventions (shared with the paper setup): each thought ends with the delimiter
``</thought>`` and the final answer is wrapped in ``\\boxed{...}``.
"""

from typing import List, Optional

THOUGHT_DELIMITER = "</thought>"

# Recommended generation / correction knobs (from the follow-up eval setup). The Thought MDP
# rollout stops on THOUGHT_DELIMITER (include it in the output) with a per-thought token budget,
# repeated up to RECOMMENDED_MAX_THOUGHTS; the correction loop runs up to
# RECOMMENDED_MAX_ICS_ITERATIONS localize -> backtrack -> resample iterations.
RECOMMENDED_MAX_THOUGHTS = 20            # max thoughts per generation
RECOMMENDED_MAX_TOKENS_PER_THOUGHT = 512  # per-thought token budget
RECOMMENDED_MAX_ICS_ITERATIONS = 10       # max correction iterations
# Sampling temperatures are left to your existing Thought-ICS settings (unchanged here).


# ---------------------------------------------------------------------------------------
# 1) Thought-by-thought generation
# ---------------------------------------------------------------------------------------

# Format guidance only (no in-context examples). Recommended default: keeps the model from
# anchoring on example phrasing, and travels better across model families.
GENERATION_PROMPT_NO_EXAMPLES = """You are solving a problem by producing one reasoning step at a time.

Do not try to solve the entire problem at once. Given the previously taken steps, think about what the single next step should be, then articulate it clearly and conclude just that step with </thought>.

Each step should be a complete, self-contained thought — one observation, calculation, or deduction that:
- Makes forward progress toward the solution
- Contains substantive reasoning (not filler like "let me think" or restating the problem)
- Coheres logically with the previous steps

When your next step arrives at the final answer, include \\boxed{{answer}} and end with </thought>.

Q: {question}
"""

# Same protocol, with two in-context formatting examples — useful for smaller models that
# need a worked demonstration of the </thought>-delimited, \\boxed{} format.
GENERATION_PROMPT_WITH_EXAMPLES = """You are solving a problem step-by-step.

Instructions:
1. State your next reasoning step (one observation, calculation, or deduction)
2. End each thought with </thought>
3. Continue until you reach the final answer, then write it in \\boxed{{answer}} format

Examples:

Q: In how many ways can 5 distinct books be arranged on a shelf if 2 specific books must not be adjacent?
Total arrangements without restrictions is 5! = 120</thought>
I need to subtract arrangements where the 2 specific books ARE adjacent</thought>
If I treat the 2 books as a single unit, I have 4 units to arrange: 4! = 24 ways</thought>
The 2 books within their unit can be arranged in 2! = 2 ways</thought>
So arrangements with the books adjacent = 24 x 2 = 48</thought>
Therefore, arrangements where they are NOT adjacent = 120 - 48 = \\boxed{{72}}</thought>

Q: A rectangle has area 48 and perimeter 28. What is the length of its diagonal?
Let length = l and width = w. From the area: lw = 48</thought>
From the perimeter: 2l + 2w = 28, so l + w = 14</thought>
From l + w = 14, we get w = 14 - l. Substituting into lw = 48: l(14 - l) = 48</thought>
Expanding: 14l - l^2 = 48, so l^2 - 14l + 48 = 0. Factoring: (l - 6)(l - 8) = 0</thought>
So l = 8 and w = 6 (or vice versa). Using the Pythagorean theorem: d^2 = 8^2 + 6^2 = 64 + 36 = 100</thought>
Therefore d = 10, so the answer is \\boxed{{10}}</thought>

Q: {question}
"""


def thought_generation_prompt(question: str, with_examples: bool = False) -> str:
    """Recommended initial prompt for thought-by-thought generation in a Thought MDP.

    The model emits ONE complete reasoning step at a time, each terminated by
    ``</thought>``, until it produces a ``\\boxed{answer}``. At each step, append the
    generated thought (including the ``</thought>`` delimiter) to the running prompt and
    sample the next step, stopping generation on ``</thought>``.

    Args:
        question: The problem statement.
        with_examples: If True, include two in-context formatting examples (helps smaller
            models follow the format); default False (recommended for capable models).
    """
    template = GENERATION_PROMPT_WITH_EXAMPLES if with_examples else GENERATION_PROMPT_NO_EXAMPLES
    return template.format(question=question)


# ---------------------------------------------------------------------------------------
# 2) Error localization
# ---------------------------------------------------------------------------------------

def localization_prompt(
    question: str,
    chain: List[str],
    ground_truth: Optional[str] = None,
) -> str:
    """Recommended self-localization prompt: identify the FIRST erroneous step.

    The model is shown its numbered reasoning chain and must return the 1-indexed step
    number of the first (originating) error as ``\\boxed{N}`` on the final line.

    Args:
        question: The problem statement.
        chain: The reasoning steps (each a string; any ``</thought>`` markers are stripped).
        ground_truth: If provided, uses the oracle (L1) framing that reveals the correct
            answer. If None (recommended default), uses the binary (L2) originating-cause
            framing — no ground truth revealed.

    Returns:
        A prompt string whose response must end with ``\\boxed{N}``.
    """
    chain_text = ""
    for i, step in enumerate(chain, 1):
        clean = step.replace(THOUGHT_DELIMITER, "").strip()
        chain_text += f"\nStep {i}: {clean}"
    n_steps = len(chain)

    if ground_truth is not None:
        # L1 (oracle): the correct answer is revealed.
        return (
            f"Problem: {question}\n\n"
            f"Current reasoning chain (WRONG - got incorrect answer):\n"
            f"{chain_text}\n\n"
            f"The correct answer should be {ground_truth}.\n\n"
            f"Analyze the reasoning chain step by step to identify where the error "
            f"occurred. Which step number (1 to {n_steps}) contains the first critical "
            f"error that led to the wrong answer?\n\n"
            f"Do NOT solve the problem again. Your ONLY task is to identify the first "
            f"erroneous step. Provide your reasoning, then put ONLY the step number (an "
            f"integer from 1 to {n_steps}) in the format: \\boxed{{step_number}}\n"
        )

    # L2 (binary feedback, recommended default): originating-cause, early-biasing framing.
    return (
        f"You are tasked with localizing the first erroneous thought in your previous "
        f"solution to this problem.\n\n"
        f"Problem: {question}\n\n"
        f"Your incorrect reasoning chain:\n{chain_text}\n\n"
        f"The final answer this chain produces is incorrect — therefore at least one step "
        f"contains an error. The error you are looking for is the originating step where a "
        f"key decision or action derailed the reasoning, not just the step where the failure "
        f"ultimately becomes visible. A misread of the problem, an unjustified assumption, "
        f"or a logical flaw can look fine for several follow-on steps before it surfaces in "
        f"the wrong answer. A step is erroneous if you cannot justify its claims from the "
        f"problem statement and earlier verified steps alone. Find the originating step, not "
        f"just the symptom.\n\n"
        f"Do NOT re-solve the problem. Your ONLY task is to identify the step number of that "
        f"originating error.\n\n"
        f"Requirements:\n"
        f"- Commit to exactly ONE step number (1 to {n_steps}).\n"
        f"- Stop at the first step you cannot justify.\n"
        f"- MANDATORY final line: your response MUST end with \\boxed{{N}} on its own line, "
        f"where N is the step index (1-indexed) of the first erroneous step in the chain "
        f"above — NOT the answer to the problem. Do NOT add any text after the \\boxed{{N}}.\n"
    )
