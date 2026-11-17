"""Prompts, parsers and budgets for Thought-ICS and Token-ICS.

Thought-ICS generates reasoning thought by thought: each generation call writes one paragraph (one
step), and the next call continues the same response from the exact tokens so far. Token-ICS
operates on the same response joined into unstructured text and localizes errors by quoting them.
Both methods use the same generation and localization budgets, sampling settings, chat templates
and prompt wording; only the unit of localization differs (a step number vs a quoted span).
"""

import re
import unicodedata
from typing import List, Optional, Tuple

# --- Budgets and sampling ---------------------------------------------------------------
# All models share a 32,768-token context (Qwen2.5-32B's limit) so that the budgets are uniform
# across models and methods.
MAX_MODEL_LEN = 32768
RESPONSE_LENGTH = 16384          # whole solution / chain, including any kept prefix on resampling
LOCALIZATION_MAX_TOKENS = 8192
VERIFY_MAX_TOKENS = 16384        # self-verification (also every Self-Refine / CoVe call)
MAX_TOKENS_PER_THOUGHT = 2048    # safety cap per thought; never more than the remaining budget
MAX_THOUGHTS = 500               # high enough that the token budget, not the step count, binds
MAX_ICS_ITERATIONS = 10
MAX_EMPTY_RETRIES = 3            # regenerate a first attempt that comes back with no thoughts
GENERATION_TEMP = 0.5
LOCALIZATION_TEMP = 0.5
TOP_P = 0.9
TOP_K = 50
# A thought ends at a paragraph break (a blank line, or a markdown hard line break).
THOUGHT_DELIMITER = "\n\n"
THOUGHT_STOPS = [THOUGHT_DELIMITER, "  \n"]


# --- Generation prompts -------------------------------------------------------------
THOUGHT_TEMPLATE = """Solve the problem step by step. Write your reasoning as a sequence of steps, one step per paragraph, with a blank line between steps.

Each step must do exactly one thing: a single observation, calculation, or deduction, usually one or two sentences. Whenever you move on to something new (a new calculation, a new case, a new deduction, the next option), start a new paragraph. Each step should:
- Make forward progress toward the solution
- Contain substantive reasoning (not filler like "let me think" or restating the problem)
- Cohere logically with the previous steps

Keep each step to a single paragraph (no blank lines inside a step), keep any equation in the same paragraph as the sentence that introduces it, and do not insert headers or separator lines (such as ---) between steps. Continue until you reach the final answer, and include it as \\boxed{{answer}} in the step that arrives at it.

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
    """The chain as shown to the localizer: '\\nStep i: <thought>'."""
    out = ""
    for i, t in enumerate(thoughts, 1):
        body = re.sub(r"\n\s*\n", "\n", t.strip())
        out += f"\nStep {i}: {body}"
    return out


def thought_localization_prompt(question: str, thoughts: List[str], verify: str = "oracle",
                                critique: Optional[str] = None) -> str:
    """Thought-ICS localization prompt: find the first erroneous step and return its number.

    verify="oracle": the model is told the final answer is incorrect. verify="self": the premise is
    the model's own verification verdict, and \\boxed{0} ("no erroneous step") is allowed. With a
    critique (Thought-ICS-F), the verification reasoning is shown after the premise."""
    if verify == "self":
        n = len(thoughts)
        return (thought_localization_prompt(question, thoughts, "oracle")
                .replace("Your incorrect reasoning chain:", "Your reasoning chain:")
                .replace("The final answer this chain produces is incorrect — therefore at least one step contains an error.",
                         "Your own check of this chain concluded that its final answer is likely incorrect — if so, at least one step contains an error." + (CRITIQUE_SECTION.format(critique=critique) if critique is not None else ""))
                .replace(f"- Commit to exactly ONE step number (1 to {n}).\n",
                         f"- Commit to exactly ONE step number (1 to {n}).\n"
                         "- If, after careful checking, no step is erroneous, the final line must instead be \\boxed{0}.\n"))
    n_steps = len(thoughts)
    ct = chain_text(thoughts)
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


def token_localization_prompt(question: str, solution: str, verify: str = "oracle",
                              critique: Optional[str] = None) -> str:
    """Token-ICS localization prompt: the Thought-ICS wording, localized by an exact quote.
    verify="self": the same changes as thought_localization_prompt, with \\boxed{NO_ERROR}."""
    if verify == "self":
        return (token_localization_prompt(question, solution, "oracle")
                .replace("Your incorrect solution:", "Your solution:")
                .replace("The final answer this solution produces is incorrect — therefore at least one part of it contains an error.",
                         "Your own check of this solution concluded that its final answer is likely incorrect — if so, at least one part of it contains an error." + (CRITIQUE_SECTION.format(critique=critique) if critique is not None else ""))
                .replace("originating error begins.\n- Stop at the first claim",
                         "originating error begins.\n"
                         "- If, after careful checking, nothing is erroneous, the final line must instead be \\boxed{NO_ERROR}.\n"
                         "- Stop at the first claim"))
    quote_fmt = '\\boxed{ERROR_QUOTE: "exact text from the solution"}'
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


# --- Carried feedback (Thought-ICS-F) ---------------------------------------------------
# The verification reasoning v is shown to the localizer; v and the localization reasoning l are
# appended to the resampling prompt. Only the current iteration's reasoning is used.
CRITIQUE_SECTION = "\n\nYour check of it:\n{critique}"
FEEDBACK_GEN_SUFFIX = "\n\nFeedback on a previous attempt at this problem:\n{critique}"


LOCALIZATION_GEN_SUFFIX = "\n\nWhere the previous attempt first went wrong:\n{localization}"


FEEDBACK_GEN_CLOSE = ("\n\nTaking this feedback into account, continue the solution: keep writing the next "
                      "steps of the solution (not a review of it), and end with the final answer as \\boxed{answer}.")


def unbox(text: str) -> str:
    """Drop \\boxed{...} wrappers (keep their content) so verdicts / step numbers in carried
    feedback are not imitated as, or graded as, the resample's answer."""
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            return "".join(out) + text[i:]
        out.append(text[i:j])
        k, depth = j + len("\\boxed{"), 1
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[j + len("\\boxed{"):k - 1] if depth == 0 else text[j + len("\\boxed{"):])
        i = k


def feedback_gen_prompt(gen_prompt_text: str, critique: str, localization: Optional[str] = None) -> str:
    """Resampling prompt with feedback: the generation prompt + the verification reasoning + the
    localization reasoning (both unboxed) + an instruction to continue the solution. The kept prefix
    of the previous attempt is then continued as the response."""
    out = gen_prompt_text + FEEDBACK_GEN_SUFFIX.format(critique=unbox(critique))
    if localization is not None:
        out += LOCALIZATION_GEN_SUFFIX.format(localization=unbox(localization))
    return out + FEEDBACK_GEN_CLOSE


def verify_prompt(question: str, solution: str, answer: str) -> str:
    """Self-Refine-style self-verification prompt: the model verifies the response with the full
    response in view. Both methods verify the same plain-text solution."""
    return f"""You are reviewing a solution to a problem. Analyze it carefully to see if they arrived at the right answer.

Problem: {question}

Solution to review:
{solution}

Final answer: {answer}

Verify the reasoning step by step and determine whether the final answer is correct or not.

Conclude with \\boxed{{YES}} if the solution is correct, or \\boxed{{NO}} if it contains errors.
MANDATORY final line: your response MUST end with \\boxed{{YES}} or \\boxed{{NO}} on its own line."""


# --- CoVe-style self-verification (the verification step of Chain-of-Verification) --------------
# (1) plan verification questions with the response in view, (2) answer each question in a separate
# call that sees only the problem, (3) judge the response against the independent answers. The
# judge's reply is the verification response (carried forward as feedback in Thought-ICS-F).
COVE_MAX_QUESTIONS = 5
COVE_PLAN_MAX_TOKENS = 2048
COVE_ANSWER_MAX_TOKENS = 2048

COVE_PLAN_PROMPT = """You are checking a step-by-step solution to a problem. Write verification questions that would reveal whether the solution is correct: each question should check one specific fact, calculation or deduction that the solution relies on, and must be answerable on its own, without seeing the solution.

Problem: {question}

Solution to check:
{steps}

Final answer: {answer}

List at most 5 questions, one per line, numbered "Q1:", "Q2:", and so on. Write only the questions."""

COVE_ANSWER_PROMPT = """Answer the following question about a problem. Work it out yourself, briefly and carefully, and state your answer clearly at the end.

Problem: {question}

Question: {q}"""

COVE_JUDGE_PROMPT = """You are checking a step-by-step solution to a problem. Each verification question below was answered independently, without seeing the solution.

Problem: {question}

Solution to check:
{steps}

Final answer: {answer}

Independent verification:
{qa}

Compare the solution with the independent answers. A disagreement on something the final answer depends on means the solution is likely wrong; check such a disagreement carefully yourself before deciding.

Conclude with \\boxed{{YES}} if the solution's final answer is correct, or \\boxed{{NO}} if it is incorrect.
MANDATORY final line: your response MUST end with \\boxed{{YES}} or \\boxed{{NO}} on its own line."""


def numbered_steps(thoughts: Optional[List[str]], solution: str) -> str:
    """The response as numbered thoughts ('Step k: ...', blank-line separated); an unstructured
    response is a single step."""
    return "\n\n".join(f"Step {k}: {t.strip()}" for k, t in enumerate(thoughts or [solution], 1))


def parse_questions(text: str, k: int = COVE_MAX_QUESTIONS) -> List[str]:
    """Verification questions from a plan: lines numbered 'Q1:', 'Q2:', ...; otherwise lines ending
    with a question mark. At most k."""
    qs = re.findall(r"^\s*\**\s*Q\s*\d+\s*[:.)]\**\s*(.+?)\s*$", text or "", flags=re.M)
    if not qs:
        qs = [ln.strip(" -*") for ln in (text or "").splitlines() if ln.strip().endswith("?")]
    return [q for q in qs if q][:k]


def cove_judge_prompt(n_tokens, max_len: int, question: str, steps: str, answer: str, qa: List[str],
                      reserve: int = 4096) -> str:
    """Judge prompt that fits the context: the independent answers are shortened (keeping their
    endings, where the answer is stated) until the prompt leaves `reserve` tokens for the judgment.
    n_tokens(text) -> prompt length in tokens."""
    keep = 4000
    while True:
        trimmed = [x if len(x) <= keep else "Q/A (shortened): ..." + x[-keep:] for x in qa]
        text = COVE_JUDGE_PROMPT.format(question=question, steps=steps, answer=answer,
                                        qa="\n\n".join(trimmed) or "(no questions)")
        if n_tokens(text) <= max_len - reserve or keep <= 200:
            return text
        keep //= 2


# --- Verdicts -------------------------------------------------------------------------------
# A reply with no parseable verdict (off-format, or cut off at the budget) is continued with this
# line and a few greedy tokens; if that still gives no verdict, the verdict counts as NO.
VERDICT_FOLLOWUP = "\n\nFinal verdict: \\boxed{"
VERDICT_FOLLOWUP_TOKENS = 4
_SPECIAL = ("<|eot_id|>", "<|im_end|>", "<|return|>", "<|end|>", "<|endoftext|>")
_YES_NO = re.compile(r"(?<![A-Za-z])(YES|NO)(?![A-Za-z])")


def parse_verdict(response: str) -> Optional[bool]:
    """True = the model says correct, False = says incorrect, None = no verdict. The last
    \\boxed{YES} / \\boxed{NO}; otherwise a final YES or NO in the last 120 characters (e.g. a
    bare 'YES' line, or 'boxed{NO}' without the backslash)."""
    b = _last_boxed(response or "")
    if b is not None:
        u = re.sub(r"\\(text|mathrm|textbf|mathbf|textsf)\s*\{([^{}]*)\}", r"\2", b).strip().strip("$ ").upper()
        if u.startswith("YES"):
            return True
        if u.startswith("NO"):
            return False
    text = response or ""
    for t in _SPECIAL:
        text = text.replace(t, "")
    hits = _YES_NO.findall(text.strip()[-120:])
    return (hits[-1] == "YES") if hits else None


def parse_followup_verdict(text: str) -> Optional[bool]:
    """Verdict from the tokens generated after VERDICT_FOLLOWUP."""
    u = text.strip().upper()
    return True if u.startswith("YES") else False if u.startswith("NO") else None


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


_STEP_LINE = re.compile(r"(?:(?:the\s+)?(?:first\s+)?(?:erroneous\s+)?(?:step|answer|final answer)\s*(?:is|:)?\s*)?"
                        r"(?:\\?boxed\s*[{.]?\s*|\*\*)?(\d+)\s*\}?\**\.?", re.I)


def parse_error_step(response: str, chain_length: int) -> Tuple[int, str]:
    """Step number from a localization response: the last \\boxed{N}; else the last "Step N";
    else a final line that is only a step number (e.g. '3', 'boxed{3}', '**3**', 'Answer: 3').

    Returns (step, status): step 0 = no erroneous step, otherwise 1-indexed. status is "boxed",
    "prose", "bare", or "default" (no valid step number: a failed localization).
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
    text = response.rsplit("<|channel|>final<|message|>", 1)[-1]
    for t in _SPECIAL:
        text = text.replace(t, "")
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    m = _STEP_LINE.fullmatch(lines[-1]) if lines else None
    if m:
        n = int(m.group(1))
        if n == 0:
            return 0, "bare"
        if 1 <= n <= chain_length:
            return n, "bare"
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
    """Quoted span from a Token-ICS localization response, with the same order as parse_error_step.

    Returns (quote, status): quote "" = no error (\\boxed{NO_ERROR}), None = no usable quote (a
    failed localization). status: "boxed" (the last \\boxed{...}, ERROR_QUOTE-tagged or a bare
    quoted span in any quote style), "prose" (the last ERROR_QUOTE: "..." elsewhere), or
    "default". Lenient about the answer format only; the quote must still match the solution
    (find_quote).
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
