"""Regression test: every prompt and parser reproduces, byte for byte, the outputs recorded from the
code that produced the paper's runs (tests/golden/golden.json)."""
import json
from pathlib import Path

from thought_ics import baselines as B
from thought_ics import ics as R
from thought_ics import protocol as P

G = json.load(open(Path(__file__).parent / "golden" / "golden.json"))
Q = "Compute $\\frac{1}{2}+x$ for x=3."
TH = ["First, note x=3.\n\n", "Then 1/2+3 = 3.5.\n\n", "So \\boxed{3.5}"]
SOL = "".join(TH)


def test_prompts_identical():
    g = G["prompts"]
    assert P.THOUGHT_TEMPLATE.format(question=Q) == g["thought_template"]
    for v in ("oracle", "self"):
        assert P.thought_localization_prompt(Q, TH, v) == g[f"thought_loc_{v}"]
        assert P.token_localization_prompt(Q, SOL, v) == g[f"token_loc_{v}"]
    assert P.thought_localization_prompt(Q, TH, "self", critique="crit \\boxed{NO}") == g["thought_loc_self_critique"]
    assert P.token_localization_prompt(Q, SOL, "self", critique="crit \\boxed{NO}") == g["token_loc_self_critique"]
    assert P.verify_prompt(Q, SOL, "3.5") == g["verify"]
    assert P.feedback_gen_prompt("GEN", "crit \\boxed{NO}", "loc \\boxed{2}") == g["feedback_gen"]
    assert P.chain_text(TH) == g["chain_text"]
    assert B.FEEDBACK_PROMPT.format(problem=Q, solution=SOL) == g["sr_feedback"]
    assert B.REFINE_PROMPT.format(problem=Q, solution=SOL, feedback="fb") == g["sr_refine"]
    assert B.PLAN_VERIFICATION_PROMPT.format(problem=Q, baseline=SOL) == g["cv_plan"]
    assert B.EXECUTE_VERIFICATION_PROMPT.format(question="q?") == g["cv_exec"]
    assert B.FINAL_RESPONSE_PROMPT.format(problem=Q, baseline=SOL, verifications="Q1: a\nA1: b\n\n") == g["cv_final"]
    steps = P.numbered_steps(TH, SOL)
    assert steps == g["cove_steps"] and P.numbered_steps(None, SOL) == g["cove_steps_unstructured"]
    assert P.COVE_PLAN_PROMPT.format(question=Q, steps=steps, answer="3.5") == g["cove_plan"]
    assert P.COVE_ANSWER_PROMPT.format(question=Q, q="What is 1/2+3?") == g["cove_answer"]
    assert P.cove_judge_prompt(len, 10**6, Q, steps, "3.5", ["Q: a?\nA: b", "Q: c?\nA: d"]) == g["cove_judge"]
    assert P.cove_judge_prompt(len, 2000 + 4096, Q, steps, "3.5", ["Q: a?\nA: " + "x" * 9000]) == g["cove_judge_short"]


def test_parsers_identical():
    g = G["parse"]
    for k, v in g.items():
        kind, _, arg = k.partition("|")
        if kind == "step":
            assert list(P.parse_error_step(arg, 3)) == v, k
        elif kind == "quote":
            assert list(P.parse_error_quote(arg)) == v, k
        elif kind == "verdict":
            assert P.parse_verdict(arg) == v, k
        elif kind == "srfb":
            assert list(B.feedback_verdict(arg)) == v, k
    for text, qs in G["parse_questions"].items():
        assert P.parse_questions(text) == qs, text
    assert P.find_quote(SOL, "Then  1/2") == g["find_quote"]
    assert P.unbox("a \\boxed{x} b") == g["unbox"]
    h = ("<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>"
         "ans \\boxed{1}<|return|>")
    assert R.final_message(h) == g["final_message"] and R.response_text(h) == g["response_text"]
    assert R.boxed_answer(SOL) == g["boxed"]
    u = G["units"]
    assert [P.has_content(x) for x in ["---", "x=1"]] == u["has_content"]
    assert [P.is_display_math(x) for x in ["\\[x=1\\]", "x"]] == u["display"]
    assert [P.opens_next(x) for x in ["We get:", "Thus", "x=1."]] == u["opens"]
