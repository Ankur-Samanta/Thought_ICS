"""Matched Thought-ICS / Token-ICS protocol for the budget-controlled study.

Thought-ICS follows SRPO (``SIERA/SRPO``, commit c99cdbb), with the thought delimiter
changed to a blank line (see THOUGHT_DELIMITER):

- budgets and temperatures: ``training/config/thought_agent_config_ics_eval.yaml``
- thought prompt:           ``training/prompt_templates.py::prompt_template_no_examples``
  (the ICS eval config uses ``use_examples: false``); the step criteria are kept verbatim,
  the delimiter instruction now asks for one step per paragraph
- generation loop rules:    ``training/thought_agent_loop.py::_generate_thoughts_core``
- localization prompt:      ``training/thought_ics_agent_loop.py::_localize_error``
- step parser:              ``training/thought_ics_agent_loop.py::_parse_error_step``

Token-ICS mirrors each Thought-ICS choice. Same generation budget, same localization
budget, temperature, chat-template handling and prompt wording, and the same parse
fallback order. The only difference is the unit of localization: a quoted span instead
of a step number.

Deviation from SRPO: SRPO encodes the generation prompt as raw text (its models, OLMo
and Qwen, were trained that way). Here every generation call is a chat-templated user
turn for both methods, which is the correct input format for Instruct and harmony
(gpt-oss) models. Thoughts / the CoT are generated inside one assistant reply, and a
reply that ends (EOS) without \\boxed{} ends the chain for both methods.
"""

import re
import unicodedata
from typing import List, Optional, Tuple

# --- Budgets / sampling (SRPO thought_agent_config_ics_eval.yaml unless noted) -------
# Depth: high enough that the 16k total token budget, not the step count, is what binds
# (paragraph steps can be ~20-30 tokens; at 100 steps 15% of gpt-oss-120b AIME chains hit the cap).
MAX_THOUGHTS = 500
# Per-thought cap is only a safety limit (a blank line defines a step). SRPO's 512 split ~3% of
# gpt-oss paragraphs mid-sentence, so it is raised (uniformly) to 2048.
MAX_TOKENS_PER_THOUGHT = 2048
# A thought is one paragraph: generation of each thought stops at a blank line, and the next
# thought is generated conditioned on the exact tokens so far (inference-time structure; no
# retroactive splitting). </thought> is not used: gpt-oss ignores it (5-32% compliance across
# prompt/format variants) and Qwen ends its chat reply after it (diag_thought_*.py), whereas
# blank-line paragraph breaks are produced reliably by all models.
THOUGHT_DELIMITER = "\n\n"
# A markdown hard line break also ends a thought: gpt-oss writes one step per line inside a
# paragraph ("...  \n...  \n"); without it ~30% of gpt-oss-20b thoughts bundle several steps
# (scripts/budget_study/diag_step_prompts.py + judge_variants.py).
THOUGHT_STOPS = [THOUGHT_DELIMITER, "  \n"]
LOCALIZATION_MAX_TOKENS = 4096
LOCALIZATION_TEMP = 0.5
MAX_ICS_ITERATIONS = 10
AUTONOMY_LEVEL = 2
# SRPO caps a whole thought chain at the rollout `response_length` (5120 in most SCPO/SRPO
# eval configs). Token-ICS gets the same total budget for the whole solution. The study
# overrides both budgets per model (run_study.MODELS) because SRPO's values bind for these
# larger models; these SRPO values remain the fallbacks.
RESPONSE_LENGTH = 5120
# Paper sampling settings (appendix "Inference"): temperature 0.5, top-p 0.9, top-k 50.
GENERATION_TEMP = 0.5
TOP_P = 0.9
TOP_K = 50
# SRPO regenerates a fresh chain when one comes back with zero thoughts. Applied to both
# methods, with a bounded number of retries.
MAX_EMPTY_RETRIES = 3


# --- Generation prompts -------------------------------------------------------------
THOUGHT_TEMPLATE = """Solve the problem step by step. Write your reasoning as a sequence of steps, one step per paragraph, with a blank line between steps.

Each step must do exactly one thing: a single observation, calculation, or deduction, usually one or two sentences. Whenever you move on to something new (a new calculation, a new case, a new deduction, the next option), start a new paragraph. Each step should:
- Make forward progress toward the solution
- Contain substantive reasoning (not filler like "let me think" or restating the problem)
- Cohere logically with the previous steps

Keep each step to a single paragraph (no blank lines inside a step), keep any equation in the same paragraph as the sentence that introduces it, and do not insert headers or separator lines (such as ---) between steps. Continue until you reach the final answer, and include it as \\boxed{{answer}} in the step that arrives at it.

Q: {question}
"""

# Token-ICS version: the same guidance without the one-step-at-a-time structure.
TOKEN_TEMPLATE = """You are solving a problem. Reason through it step by step, then give the final answer.

Your reasoning should:
- Make forward progress toward the solution
- Contain substantive reasoning (not filler like "let me think" or restating the problem)
- Cohere logically from one step to the next

Continue until you reach the final answer, and include it as \\boxed{{answer}}.

Q: {question}
"""


# --- Localization prompts -----------------------------------------------------------
def has_content(unit: str) -> bool:
    """A generated paragraph counts as a thought only if it has reasoning content (any letter
    or digit); separator/whitespace-only units (e.g. '---') are attached to the previous
    thought. Tokens are unchanged -- this only decides where a step boundary is recorded."""
    return re.search(r"[A-Za-z0-9]", unit) is not None


_DISPLAY_MATH = re.compile(r"\s*(\\\[.*\\\]|\$\$.*\$\$|\\begin\{[a-zA-Z*]+\}.*\\end\{[a-zA-Z*]+\})\s*[.,;]?\s*",
                           re.DOTALL)


_TRAILING_SEPARATORS = re.compile(r"(\s*^[ \t]*([-*_])\2{2,}[ \t]*$)+\s*\Z", re.MULTILINE)
_CONNECTIVES = {"thus", "hence", "so", "therefore", "then", "now", "next", "finally", "consequently"}


def is_display_math(unit: str) -> bool:
    """A paragraph that is only a displayed equation (\\[...\\], $$...$$, \\begin{..}..\\end{..}),
    ignoring trailing separator lines. gpt-oss puts display math in its own paragraph
    ("... gives" / "\\[ b+3c=11 \\]"); such a unit completes the previous thought rather than
    being a reasoning step of its own."""
    return _DISPLAY_MATH.fullmatch(_TRAILING_SEPARATORS.sub("", unit)) is not None


_HEADER = re.compile(r"(#{1,6}\s+.*|\*\*[^*]+\*\*[.:]?)")
_DANGLING = re.compile(r"(=|,|;|\b(is|are|was|were|be|gives|give|yields|becomes|equals|so|thus|then|"
                       r"and|or|of|to|as|by|with|from|that|which|where|for|in|than|get|obtain))$",
                       re.IGNORECASE)


def opens_next(thought: str) -> bool:
    """A thought that is unfinished on its own gets the next unit attached to it:
    it introduces what follows (ends with ':'), is a bare connective ('Thus'), is only a
    markdown header ('**Case k=4.**'), or ends mid-sentence on an operator / connecting word
    ('Thus the equation is', 'so that x ='). Lines that merely lack a period (list items) are
    left alone."""
    t = _TRAILING_SEPARATORS.sub("", thought).strip()
    if not t:
        return False
    return (t.endswith(":") or t.rstrip(".,").lower() in _CONNECTIVES
            or _HEADER.fullmatch(t) is not None or _DANGLING.search(t) is not None)


def chain_text(thoughts: List[str]) -> str:
    """SRPO formatting: '\\nStep i: <thought without delimiter>'."""
    out = ""
    for i, t in enumerate(thoughts, 1):
        out += f"\nStep {i}: {re.sub(r'\n\s*\n', chr(10), t.strip())}"
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


_QUOTE_CLOSERS = {'"': '"', "\u201c": "\u201d", "'": "'", "\u2018": "\u2019", "`": "`"}


def _quoted(body: str, boxed: bool) -> Optional[str]:
    """Text inside a leading quote of any common style; None if ``body`` is not quoted."""
    body = body.strip()
    if not body or body[0] not in _QUOTE_CLOSERS:
        return None
    closer, inner = _QUOTE_CLOSERS[body[0]], body[1:]
    k = inner.rfind(closer) if boxed else inner.find(closer + "}")
    if k == -1:
        k = inner.rfind(closer)
    q = inner[:k].strip() if k > 0 else ""
    return q or None


def _after_last_tag(text: str) -> Optional[str]:
    tags = list(re.finditer(r"ERROR_QUOTE\s*:\s*", text, re.IGNORECASE))
    return text[tags[-1].end():] if tags else None


def parse_error_quote(response: str) -> Tuple[Optional[str], str]:
    """Token-ICS version of ``parse_error_step``, with the same fallback order.

    Returns (quote, status): quote "" = no error (\\boxed{NO_ERROR}), None = no usable
    quote, which maps to SRPO's "default to step 1" (resample from an empty prefix).
    status: "boxed" (last \\boxed{...}: ERROR_QUOTE-tagged or a bare quoted span, any quote
    style), "prose" (last ERROR_QUOTE: "..." elsewhere, e.g. a box with unbalanced braces),
    or "default". Lenient about the answer FORMAT only; the quote itself must still match
    the solution (see find_quote).
    """
    boxed = _last_boxed(response)
    if boxed is not None:
        if boxed.strip().upper() in ("NO_ERROR", "CORRECT"):
            return "", "boxed"
        rest = _after_last_tag(boxed)
        q = _quoted(rest if rest is not None else boxed, boxed=True)
        if q:
            return q, "boxed"
    rest = _after_last_tag(response)
    if rest is not None:
        q = _quoted(rest, boxed=False)
        if q:
            return q, "prose"
    return None, "default"


def _normalize_with_map(text: str):
    """NFKC + collapse whitespace runs to one space; return (normalized, map to original idx)."""
    out, idx_map, prev_space = [], [], False
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            if c.isspace():
                if prev_space:
                    continue
                c, prev_space = " ", True
            else:
                prev_space = False
            out.append(c)
            idx_map.append(i)
    return "".join(out), idx_map


def find_quote(solution: str, quote: str) -> Optional[int]:
    """Location of the quote in the solution, or None.

    Exact match, up to whitespace and Unicode normalization only (e.g. gpt-oss writes
    U+202F thin spaces). Paraphrased, drifting or invented quotes still fail: copying the
    span faithfully is part of Token-ICS localization.
    """
    idx = solution.find(quote)
    if idx != -1:
        return idx
    norm_sol, idx_map = _normalize_with_map(solution)
    norm_q = _normalize_with_map(quote)[0].strip()
    if not norm_q:
        return None
    j = norm_sol.find(norm_q)
    return idx_map[j] if j != -1 else None
