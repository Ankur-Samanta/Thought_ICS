"""Matched Thought-ICS / Token-ICS protocol for the budget-controlled study.

Thought-ICS is copied from SRPO (``SIERA/SRPO``, commit c99cdbb):

- budgets and temperatures: ``training/config/thought_agent_config_ics_eval.yaml``
- thought prompt:           ``training/prompt_templates.py::prompt_template_no_examples``
  (the ICS eval config uses ``use_examples: false``)
- generation loop rules:    ``training/thought_agent_loop.py::_generate_thoughts_core``
- localization prompt:      ``training/thought_ics_agent_loop.py::_localize_error``
- step parser:              ``training/thought_ics_agent_loop.py::_parse_error_step``

Token-ICS mirrors each Thought-ICS choice. Same generation budget, same localization
budget, temperature, chat-template handling and prompt wording, and the same parse
fallback order. The only difference is the unit of localization: a quoted span instead
of a step number.
"""

import re
from typing import List, Optional, Tuple

# --- Budgets / sampling (SRPO thought_agent_config_ics_eval.yaml unless noted) -------
MAX_THOUGHTS = 20
MAX_TOKENS_PER_THOUGHT = 512
THOUGHT_DELIMITER = "</thought>"
LOCALIZATION_MAX_TOKENS = 4096
LOCALIZATION_TEMP = 0.5
MAX_ICS_ITERATIONS = 10
AUTONOMY_LEVEL = 2
# SRPO caps a whole thought chain at the rollout `response_length`. Most SCPO/SRPO eval
# configs use 5120. Token-ICS gets the same total budget for the whole solution.
RESPONSE_LENGTH = 5120
# Paper sampling settings (appendix "Inference"): temperature 0.5, top-p 0.9, top-k 50.
GENERATION_TEMP = 0.5
TOP_P = 0.9
TOP_K = 50
# SRPO regenerates a fresh chain when one comes back with zero thoughts. Applied to both
# methods, with a bounded number of retries.
MAX_EMPTY_RETRIES = 3


# --- Generation prompts -------------------------------------------------------------
THOUGHT_TEMPLATE = """You are solving a problem by producing one reasoning step at a time.

Do not try to solve the entire problem at once. Given the previously taken steps, think about what the single next step should be, then articulate it clearly and conclude just that step with </thought>.

Each step should be a complete, self-contained thought — one observation, calculation, or deduction that:
- Makes forward progress toward the solution
- Contains substantive reasoning (not filler like "let me think" or restating the problem)
- Coheres logically with the previous steps

When your next step arrives at the final answer, include \\boxed{{answer}} and end with </thought>.

Q: {question}
"""

# Token-ICS version: the same guidance without the one-step-at-a-time structure.
TOKEN_TEMPLATE = """You are solving a problem. Reason through it step by step, then give the final answer.

Your reasoning should:
- Make forward progress toward the solution
- Contain substantive reasoning (not filler like "let me think" or restating the problem)
- Cohere logically from one step to the next

When you arrive at the final answer, include \\boxed{{answer}}.

Q: {question}
"""


# --- Localization prompts -----------------------------------------------------------
def chain_text(thoughts: List[str]) -> str:
    """SRPO formatting: '\\nStep i: <thought without delimiter>'."""
    out = ""
    for i, t in enumerate(thoughts, 1):
        out += f"\nStep {i}: {t.replace(THOUGHT_DELIMITER, '').strip()}"
    return out


def thought_localization_prompt(question: str, thoughts: List[str], ground_truth: str,
                                autonomy_level: int = AUTONOMY_LEVEL) -> str:
    """Verbatim copy of SRPO ``_localize_error`` prompt text."""
    n_steps = len(thoughts)
    ct = chain_text(thoughts)
    if autonomy_level == 1:
        return (
            f"Problem: {question}\n\n"
            f"Current reasoning chain (WRONG - got incorrect answer):\n"
            f"{ct}\n\n"
            f"The correct answer should be {ground_truth}.\n\n"
            f"Analyze the reasoning chain step by step to identify where "
            f"the error occurred. Which step number (1 to {n_steps}) "
            f"contains the first critical error that led to the wrong "
            f"answer?\n\n"
            f"Do NOT solve the problem again. Your ONLY task is to "
            f"identify the first erroneous step. Provide your reasoning, "
            f"then put ONLY the step number (an integer from 1 to "
            f"{n_steps}) in the format: \\boxed{{step_number}}\n"
        )
    return (
        f"You are tasked with localizing the first erroneous "
        f"thought in your previous solution to this problem.\n\n"
        f"Problem: {question}\n\n"
        f"Your incorrect reasoning chain:\n{ct}\n\n"
        f"The final answer this chain produces is incorrect — "
        f"therefore at least one step contains an error. The error "
        f"you are looking for is the originating step where a key "
        f"decision or action derailed the reasoning, not just the "
        f"step where the failure ultimately becomes visible. A "
        f"misread of the problem, an unjustified assumption, or a "
        f"logical flaw can look fine for several follow-on steps "
        f"before it surfaces in the wrong answer. A step is "
        f"erroneous if you cannot justify its claims from the "
        f"problem statement and earlier verified steps alone. "
        f"Find the originating step, not just the symptom.\n\n"
        f"Do NOT re-solve the problem. Your ONLY task is to "
        f"identify the step number of that originating error.\n\n"
        f"Requirements:\n"
        f"- Commit to exactly ONE step number (1 to {n_steps}).\n"
        f"- Stop at the first step you cannot justify.\n"
        f"- MANDATORY final line: your response MUST end with "
        f"\\boxed{{N}} on its own line, where N is the step index "
        f"(1-indexed) of the first erroneous step in the chain "
        f"above — NOT the answer to the problem. Do NOT add any "
        f"text after the \\boxed{{N}}.\n"
    )


def token_localization_prompt(question: str, solution: str, ground_truth: str,
                              autonomy_level: int = AUTONOMY_LEVEL) -> str:
    """Token-ICS version of the SRPO prompt: same wording, localized by quote."""
    quote_fmt = '\\boxed{ERROR_QUOTE: "exact text from the solution"}'
    if autonomy_level == 1:
        return (
            f"Problem: {question}\n\n"
            f"Current solution (WRONG - got incorrect answer):\n"
            f"{solution}\n\n"
            f"The correct answer should be {ground_truth}.\n\n"
            f"Analyze the solution step by step to identify where "
            f"the error occurred. Where does the first critical error "
            f"that led to the wrong answer begin?\n\n"
            f"Do NOT solve the problem again. Your ONLY task is to "
            f"identify the first error. Provide your reasoning, "
            f"then put ONLY the EXACT text (word-for-word, a continuous "
            f"excerpt from the solution above) where the error begins in "
            f"the format: {quote_fmt}\n"
        )
    return (
        f"You are tasked with localizing the first error "
        f"in your previous solution to this problem.\n\n"
        f"Problem: {question}\n\n"
        f"Your incorrect solution:\n{solution}\n\n"
        f"The final answer this solution produces is incorrect — "
        f"therefore at least one part of it contains an error. The error "
        f"you are looking for is the originating point where a key "
        f"decision or action derailed the reasoning, not just the "
        f"point where the failure ultimately becomes visible. A "
        f"misread of the problem, an unjustified assumption, or a "
        f"logical flaw can look fine for several follow-on steps "
        f"before it surfaces in the wrong answer. A claim is "
        f"erroneous if you cannot justify it from the "
        f"problem statement and earlier verified reasoning alone. "
        f"Find the originating error, not just the symptom.\n\n"
        f"Do NOT re-solve the problem. Your ONLY task is to "
        f"identify where that originating error begins.\n\n"
        f"Requirements:\n"
        f"- Commit to exactly ONE location: the EXACT text (word-for-word, "
        f"a continuous excerpt from the solution above) where the "
        f"originating error begins.\n"
        f"- Stop at the first claim you cannot justify.\n"
        f"- MANDATORY final line: your response MUST end with "
        f"{quote_fmt} on its own line — NOT the answer to the "
        f"problem. Do NOT add any text after the \\boxed{{...}}.\n"
    )


# --- Parsing ------------------------------------------------------------------------
def _last_boxed(text: str) -> Optional[str]:
    """Content of the last brace-balanced \\boxed{...}, or None."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


def parse_error_step(response: str, chain_length: int) -> Tuple[int, str]:
    """SRPO ``_parse_error_step``, also returning which rule fired.

    Returns (step, status): step 0 = no error, otherwise 1-indexed.
    status is "boxed", "prose" (last "Step N" phrase) or "default" (step 1).
    """
    boxed = _last_boxed(response)
    if boxed is not None:
        try:
            n = int(boxed)
            if n == 0:
                return 0, "boxed"
            if 1 <= n <= chain_length:
                return n, "boxed"
        except (ValueError, TypeError):
            pass
    steps = list(re.finditer(r"[Ss]tep\s+(\d+)", response))
    if steps:
        n = int(steps[-1].group(1))
        if 1 <= n <= chain_length:
            return n, "prose"
    return 1, "default"


_QUOTE_RE = re.compile(r'ERROR_QUOTE:\s*["\u201c](.+?)["\u201d]\s*$', re.DOTALL)
_PROSE_QUOTE_RE = re.compile(r'ERROR_QUOTE:\s*["\u201c](.+?)["\u201d]', re.DOTALL)


def parse_error_quote(response: str) -> Tuple[Optional[str], str]:
    """Token-ICS version of ``parse_error_step``, with the same fallback order.

    Returns (quote, status): quote "" = no error (\\boxed{NO_ERROR}), None = no usable
    quote, which maps to SRPO's "default to step 1" (resample from an empty prefix).
    status is "boxed", "prose" (last ERROR_QUOTE: "..." outside a box) or "default".
    """
    boxed = _last_boxed(response)
    if boxed is not None:
        if boxed.strip().upper() in ("NO_ERROR", "CORRECT"):
            return "", "boxed"
        m = _QUOTE_RE.search(boxed)
        if m and m.group(1).strip():
            return m.group(1).strip(), "boxed"
    prose = list(_PROSE_QUOTE_RE.finditer(response))
    if prose and prose[-1].group(1).strip():
        return prose[-1].group(1).strip(), "prose"
    return None, "default"


def find_quote(solution: str, quote: str) -> Optional[int]:
    """Exact-match location of the quote (paper Token-ICS behavior). None if absent."""
    idx = solution.find(quote)
    return idx if idx != -1 else None
