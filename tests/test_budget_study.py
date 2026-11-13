"""CPU smoke test for thought_ics.budget_study with a scripted fake engine (no vLLM)."""

import json
import types

from thought_ics.budget_study import protocol as P
from thought_ics.budget_study import run_study as R


class FakeTok:
    """Character-level 'tokenizer': one id per character."""
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeEngine:
    """Scripted model.

    Thought gen: chains of 3 thoughts; the 3rd boxes 7 (wrong) unless the prompt already
    contains 'step 1 fixed', in which case it boxes 42 (right). A resample from step 2
    writes 'step 1 fixed'. The first call for problem 'p_empty' returns no thought.
    Token gen: boxes 7 unless the prefix was cut; after a cut it boxes 42.
    Localization: thought -> \\boxed{2} (or prose 'Step 1' for p_prose); token -> quote.
    """
    def __init__(self):
        self.tok = FakeTok()
        self.max_model_len = 10**6
        self.empty_served = False
        self.calls = []

    def gen_prompt_ids(self, text):
        return self.tok.encode("<gen>" + text)

    def chat_ids(self, text):
        return self.tok.encode("<chat>" + text)

    def generate(self, prompt_ids, params):
        outs = []
        self.last_prompts = getattr(self, "last_prompts", [])
        for ids, prm in zip(prompt_ids, params):
            text = self.tok.decode(ids)
            self.last_prompts.append(text)
            self.calls.append((text.startswith("<chat>"), prm.max_tokens, prm.stop))
            if text.startswith("<chat>"):
                if "first erroneous thought" in text:
                    out = "I think Step 1 is wrong" if "p_prose" in text else "reasoning\n\\boxed{2}"
                else:
                    out = 'reasoning\n\\boxed{ERROR_QUOTE: "wrong part"}'
            elif prm.stop and "\\boxed{" in text.split("Q:")[-1]:
                out = ""  # reply ends after the answer (EOS)
            elif prm.stop:  # thought generation
                n_done = text.count(P.THOUGHT_DELIMITER) - text.split("Q:")[0].count(P.THOUGHT_DELIMITER)
                if "p_empty" in text and not self.empty_served:
                    self.empty_served = True
                    out = ""
                elif n_done == 0:
                    out = "step 0 ok\n\n"
                elif n_done == 1:
                    out = "step 1 fixed\n\n" if "RESAMPLED" in text else "step 1 bad\n\n"
                else:
                    ans = 42 if "step 1 fixed" in text else 7
                    out = f"so \\boxed{{{ans}}}\n\n"
            else:  # token generation
                if text.rstrip().endswith("good part"):
                    out = " then fixed so \\boxed{42}"
                else:
                    out = "good part wrong part so \\boxed{7}"
            outs.append({"ids": self.tok.encode(out) or [1], "text": out,
                         "finish_reason": "stop",
                         "stop_reason": P.THOUGHT_DELIMITER if out.endswith(P.THOUGHT_DELIMITER) else None})
        return outs


def _cfg(**kw):
    base = dict(max_thoughts=P.MAX_THOUGHTS, max_tokens_per_thought=P.MAX_TOKENS_PER_THOUGHT,
                response_length=10**5, localization_max_tokens=P.LOCALIZATION_MAX_TOKENS,
                localization_temp=P.LOCALIZATION_TEMP, generation_temp=P.GENERATION_TEMP,
                top_p=P.TOP_P, top_k=P.TOP_K, max_ics_iterations=3,
                max_empty_retries=P.MAX_EMPTY_RETRIES, autonomy_level=2,
                token_prompt="token", token_init_from=None, fallback="scratch",
                verify_max_tokens=12288)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _patch_resample_marker(monkeypatch):
    """Make thought resamples from a prefix visible to the fake model."""
    orig = R.gen_thought_chains

    def wrapped(eng, cfg, jobs):
        for j in jobs:
            if "prefix_thoughts" in j:
                j["prompt_ids"] = j["prompt_ids"] + eng.tok.encode("RESAMPLED")
        return orig(eng, cfg, jobs)
    monkeypatch.setattr(R, "gen_thought_chains", wrapped)


PROBLEMS = [{"problem": "p_main", "answer": "42", "unique_id": "a"},
            {"problem": "p_empty", "answer": "42", "unique_id": "b"},
            {"problem": "p_prose", "answer": "42", "unique_id": "c"}]


def test_thought_ics_flow(monkeypatch):
    _patch_resample_marker(monkeypatch)
    eng = FakeEngine()
    recs = R.run_method(eng, _cfg(), "thought", PROBLEMS)
    a, b, c = recs
    assert a["iterations"][0]["answer"] == "7" and not a["iterations"][0]["correct"]
    loc = a["iterations"][1]["localization"]
    assert loc["parse_status"] == "boxed" and loc["error_step"] == 2 and loc["prefix_len"] == 1
    assert a["iterations"][1]["correct"] and a["final_correct"]
    assert b["empty_retries"] == 1 and b["final_correct"]
    assert c["iterations"][1]["localization"]["parse_status"] == "prose"
    assert c["iterations"][1]["localization"]["error_step"] == 1
    # every thought call uses the per-thought cap and only the delimiter as stop
    assert all(m == P.MAX_TOKENS_PER_THOUGHT and s == P.THOUGHT_STOPS
               for _, m, s in eng.calls if s)
    # resample continues from the exact kept token ids (inference-time structure)
    assert a["iterations"][1]["chain"][0] == a["iterations"][0]["chain"][0]
    s = R.summarize("thought", recs)
    assert s["init_accuracy"] == 0 and s["final_accuracy"] == 1


def test_token_ics_flow():
    eng = FakeEngine()
    recs = R.run_method(eng, _cfg(response_length=5000), "token", PROBLEMS[:1])
    r = recs[0]
    assert not r["iterations"][0]["correct"]
    loc = r["iterations"][1]["localization"]
    assert loc["parse_status"] == "boxed" and loc["error_quote"] == "wrong part" and loc["quote_found"]
    assert r["final_correct"]
    # generation budget = response_length minus prefix; localization budget = LOCALIZATION_MAX_TOKENS
    gen_calls = [m for chat, m, s in eng.calls if not s and not chat]
    assert gen_calls[0] == 5000 and gen_calls[1] == 5000 - len("good part")
    assert [m for chat, m, _ in eng.calls if chat] == [P.LOCALIZATION_MAX_TOKENS]


def test_thought_budget_matches_token_budget():
    """Thoughts may use whatever remains of response_length (same total as Token-ICS)."""
    eng = FakeEngine()
    first = len("step 0 ok\n\n")
    cfg = _cfg(response_length=first + 4, max_ics_iterations=0)
    g = R.gen_thought_chains(eng, cfg, [{"prompt_ids": eng.gen_prompt_ids("Q: p_main")}])[0]
    thought_calls = [m for _, m, s in eng.calls if s]
    assert thought_calls[:2] == [first + 4, 4]           # 2nd thought gets the remaining 4
    assert g["status"] == "budget" and g["n_tokens"] >= cfg.response_length


def test_parsers():
    assert P.parse_error_step("x \\boxed{3}", 5) == (3, "boxed")
    assert P.parse_error_step("x \\boxed{9}", 5) == (1, "default")
    assert P.parse_error_step("error in Step 4 here", 5) == (4, "prose")
    assert P.parse_error_step("no idea", 5) == (1, "default")
    assert P.parse_error_quote('\\boxed{ERROR_QUOTE: "a b c"}') == ("a b c", "boxed")
    assert P.parse_error_quote("\\boxed{NO_ERROR}") == ("", "boxed")
    assert P.parse_error_quote('so ERROR_QUOTE: "x y" maybe') == ("x y", "prose")
    assert P.parse_error_quote("nothing") == (None, "default")


def test_thought_eos_ends_chain():
    """A reply that ends (EOS, no stop string) without \\boxed ends the chain."""
    class EosEngine(FakeEngine):
        def generate(self, prompt_ids, params):
            return [{"ids": [1, 2], "text": "no answer here", "finish_reason": "stop",
                     "stop_reason": None} for _ in prompt_ids]
    eng = EosEngine()
    g = R.gen_thought_chains(eng, _cfg(), [{"prompt_ids": eng.gen_prompt_ids("Q: x")}])[0]
    assert g["status"] == "eos_no_answer" and len(g["thoughts"]) == 1


def test_find_quote_normalization():
    sol = "First we note 12\u202f000 / 30 = 400.  Then   we add."
    assert P.find_quote(sol, "12 000 / 30") == sol.index("12")       # Unicode space
    assert P.find_quote(sol, "Then we add") == sol.index("Then")      # whitespace runs
    assert P.find_quote(sol, "Then we multiply") is None              # misquote still fails


def test_boxed_split_across_thoughts():
    class SplitEngine(FakeEngine):
        def generate(self, prompt_ids, params):
            text = self.tok.decode(prompt_ids[0])
            if "{42}" in text:
                return [{"ids": [1], "text": "", "finish_reason": "stop", "stop_reason": None}]
            out = "so \\boxed" if not text.endswith("\\boxed") else "{42}"
            return [{"ids": self.tok.encode(out), "text": out, "finish_reason": "length",
                     "stop_reason": None} for _ in prompt_ids]
    eng = SplitEngine()
    g = R.gen_thought_chains(eng, _cfg(), [{"prompt_ids": eng.gen_prompt_ids("Q: x")}])[0]
    assert g["status"] == "answer" and R.boxed_answer(g["text"]) == "42"


def test_separator_units_merge_into_previous_thought():
    class SepEngine(FakeEngine):
        def generate(self, prompt_ids, params):
            text = self.tok.decode(prompt_ids[0])
            n = text.count("step") - text.split("Q:")[0].count("step")
            k = min(n + text.count("---"), 2)
            out = ["step one\n\n", "---\n\n", "step two \\boxed{1}"][k]
            return [{"ids": self.tok.encode(out), "text": out, "finish_reason": "stop",
                     "stop_reason": "\n\n" if k < 2 else None} for _ in prompt_ids]
    eng = SepEngine()
    g = R.gen_thought_chains(eng, _cfg(), [{"prompt_ids": eng.gen_prompt_ids("Q: x")}])[0]
    assert g["thoughts"] == ["step one\n\n---\n\n", "step two \\boxed{1}"]
    assert g["bounds"][0][1] == g["bounds"][1][0] and g["status"] == "answer"


def test_thought_chain_continues_after_box_until_reply_ends():
    """Like Token-ICS: a later revision after a first \\boxed{} is kept and graded."""
    class ReviseEngine(FakeEngine):
        def generate(self, prompt_ids, params):
            text = self.tok.decode(prompt_ids[0]).split("Q:")[-1]
            if "\\boxed{7}" not in text:
                out, sr = "so \\boxed{7}\n\n", "\n\n"
            elif "recheck" not in text:
                out, sr = "recheck: actually \\boxed{42}", 200002  # int EOS stop_reason
            else:
                raise AssertionError("generated past the end of the reply")
            return [{"ids": self.tok.encode(out), "text": out, "finish_reason": "stop",
                     "stop_reason": sr}]
    eng = ReviseEngine()
    g = R.gen_thought_chains(eng, _cfg(), [{"prompt_ids": eng.gen_prompt_ids("Q: x")}])[0]
    assert g["status"] == "answer" and R.boxed_answer(g["text"]) == "42" and len(g["thoughts"]) == 2


def test_immediate_eos_is_empty_and_retried():
    class EmptyEngine(FakeEngine):
        def generate(self, prompt_ids, params):
            return [{"ids": [1], "text": "", "finish_reason": "stop", "stop_reason": None}
                    for _ in prompt_ids]
    eng = EmptyEngine()
    recs = R.run_method(eng, _cfg(max_ics_iterations=0), "thought", PROBLEMS[:1])
    assert recs[0]["empty_retries"] == P.MAX_EMPTY_RETRIES


def test_quote_parser_format_leniency():
    assert P.parse_error_quote('\\boxed{"a b c"}') == ("a b c", "boxed")
    assert P.parse_error_quote("\\boxed{ERROR_QUOTE: 'x y'}") == ("x y", "boxed")
    assert P.parse_error_quote('end \\boxed{ERROR_QUOTE: "\\frac{a"}') == ("\\frac{a", "prose")
    assert P.parse_error_quote('\\boxed{ERROR_QUOTE: "he said "hi" here"}') == ('he said "hi" here', "boxed")
    assert P.parse_error_quote("\\boxed{42}") == (None, "default")


def test_token_shared_initial_cache(tmp_path):
    """Token-ICS starts from the joined Thought-ICS chain and resamples with the same prompt."""
    chain = ["good part ", "wrong part so \\boxed{7}"]
    rec = {"problem_id": "a", "empty_retries": 0,
           "iterations": [{"iteration": 0, "chain": chain, "gen_status": "answer"}]}
    (tmp_path / "thought_results.jsonl").write_text(json.dumps(rec) + "\n")
    eng = FakeEngine()
    cfg = _cfg(response_length=5000, token_prompt="thought", token_init_from=str(tmp_path))
    r = R.run_method(eng, cfg, "token", PROBLEMS[:1])[0]
    assert r["iterations"][0]["solution"] == "".join(chain) and not r["iterations"][0]["correct"]
    gen_prompts = [eng.last_prompts[k] for k in range(len(eng.last_prompts)) if not eng.last_prompts[k].startswith("<chat>")]
    assert all("one step per paragraph" in p for p in gen_prompts)   # Thought-ICS prompt
    assert r["final_correct"]


def test_display_math_attaches_to_previous_thought():
    assert P.is_display_math("\\[\nb+3c=11.\n\\]\n\n") and P.is_display_math("$$x=1$$")
    assert not P.is_display_math("Thus \\[x=1\\] holds.")
    class MathEngine(FakeEngine):
        def generate(self, prompt_ids, params):
            text = self.tok.decode(prompt_ids[0]).split("Q:")[-1]
            seq = ["Subtracting gives\n\n", "\\[\nb+3c=11.\n\\]\n\n", "so \\boxed{11}"]
            k = sum(x in text for x in ("gives", "b+3c"))
            sr = "\n\n" if k < 2 else None
            return [{"ids": self.tok.encode(seq[k]), "text": seq[k], "finish_reason": "stop",
                     "stop_reason": sr}]
    eng = MathEngine()
    g = R.gen_thought_chains(eng, _cfg(), [{"prompt_ids": eng.gen_prompt_ids("Q: x")}])[0]
    assert len(g["thoughts"]) == 2 and "b+3c" in g["thoughts"][0] and g["status"] == "answer"


def test_failed_localization_uses_up_attempt():
    """fallback='attempt': unusable localization -> no resample; the next iteration re-localizes."""
    class BadLocEngine(FakeEngine):
        def generate(self, prompt_ids, params):
            text = self.tok.decode(prompt_ids[0])
            if text.startswith("<chat>"):
                self.n_loc = getattr(self, "n_loc", 0) + 1
                out = "no idea" if self.n_loc == 1 else 'ok \\boxed{ERROR_QUOTE: "wrong part"}'
                return [{"ids": self.tok.encode(out), "text": out, "finish_reason": "stop",
                         "stop_reason": None}]
            return super().generate(prompt_ids, params)
    eng = BadLocEngine()
    r = R.run_method(eng, _cfg(response_length=5000, fallback="attempt", max_ics_iterations=3),
                     "token", PROBLEMS[:1])[0]
    its = r["iterations"]
    assert its[1].get("failed_localization") and "gen_status" not in its[1]
    assert its[2]["localization"]["quote_found"] and its[2]["correct"] and r["final_correct"]


def test_quote_not_found_uses_up_attempt():
    class MissEngine(FakeEngine):
        def generate(self, prompt_ids, params):
            text = self.tok.decode(prompt_ids[0])
            if text.startswith("<chat>"):
                out = '\\boxed{ERROR_QUOTE: "text that is not there"}'
                return [{"ids": self.tok.encode(out), "text": out, "finish_reason": "stop",
                         "stop_reason": None}]
            return super().generate(prompt_ids, params)
    eng = MissEngine()
    r = R.run_method(eng, _cfg(response_length=5000, fallback="attempt", max_ics_iterations=2),
                     "token", PROBLEMS[:1])[0]
    assert all(it.get("failed_localization") for it in r["iterations"][1:]) and not r["final_correct"]


def test_boundary_rules():
    assert P.is_display_math("\\[\nS_n = 2S_{n-1}\n\\]\n\n---\n\n")
    assert P.opens_next("Now count ordered pairs:\n\n") and P.opens_next("Thus\n\n")
    assert not P.opens_next("Thus the answer is 4.\n\n")
    class LeadEngine(FakeEngine):
        def generate(self, prompt_ids, params):
            text = self.tok.decode(prompt_ids[0]).split("Q:")[-1]
            seq = ["Count the pairs:\n\n", "- (1,2) and (2,1), 2 pairs\n\n",
                   "Thus\n\n", "there are 2 pairs, \\boxed{2}"]
            k = sum(x in text for x in ("Count", "(1,2)", "Thus"))
            return [{"ids": self.tok.encode(seq[k]), "text": seq[k], "finish_reason": "stop",
                     "stop_reason": "\n\n" if k < 3 else None}]
    eng = LeadEngine()
    g = R.gen_thought_chains(eng, _cfg(), [{"prompt_ids": eng.gen_prompt_ids("Q: x")}])[0]
    assert len(g["thoughts"]) == 2 and g["thoughts"][1].startswith("Thus") and g["status"] == "answer"


def test_opens_next_headers_and_unfinished_sentences():
    assert P.opens_next("**Case \\(k=4\\).**\n\n") and P.opens_next("## Step 2\n\n")
    assert P.opens_next("Thus the equation is  \n")
    assert P.opens_next("Substituting, the equation becomes  \n") and P.opens_next("so that x =  \n")
    assert not P.opens_next("- (1,2) and (2,1), 2 pairs\n\n")      # list item without a period
    assert not P.opens_next("Use \\(xy=20\\) to find \\(x^2\\).  \n")
    assert not P.opens_next("so \\(a=3\\)\\]\n\n") and not P.opens_next("The answer is \\(\\boxed{4}\\)")


class L3Engine(FakeEngine):
    """Verifier: YES iff the reviewed solution boxes 42. Localizer: step 2 / quote 'wrong part'
    (or 'no error' when self.loc_no_error)."""
    loc_no_error = False

    def generate(self, prompt_ids, params):
        outs, rest = [], []
        for ids, prm in zip(prompt_ids, params):
            text = self.tok.decode(ids)
            if text.startswith("<chat>You are reviewing"):
                out = "\\boxed{YES}" if "\\boxed{42}" in text else "\\boxed{NO}"
                outs.append({"ids": self.tok.encode(out), "text": out, "finish_reason": "stop", "stop_reason": None})
            elif text.startswith("<chat>") and self.loc_no_error:
                out = "\\boxed{0}" if "first erroneous thought" in text else "\\boxed{NO_ERROR}"
                outs.append({"ids": self.tok.encode(out), "text": out, "finish_reason": "stop", "stop_reason": None})
            else:
                outs.append(None); rest.append((ids, prm))
        sub = iter(super().generate([r[0] for r in rest], [r[1] for r in rest]) if rest else [])
        return [o if o is not None else next(sub) for o in outs]


def test_l3_self_verified_correction(monkeypatch):
    _patch_resample_marker(monkeypatch)
    eng = L3Engine()
    r = R.run_method(eng, _cfg(autonomy_level=3, fallback="attempt"), "thought", PROBLEMS[:1])[0]
    its = r["iterations"]
    assert not its[0]["correct"]
    assert its[1]["localization"]["verification"]["verdict"] is False and its[1]["correct"]
    assert its[2].get("stopped") == "verified" and r["exit"] == "verified"
    assert r["final_correct"] and r["final_correct_A"]


def test_l3_vl_disagreement_resets_to_initial():
    eng = L3Engine()
    eng.loc_no_error = True
    r = R.run_method(eng, _cfg(autonomy_level=3, fallback="attempt", response_length=5000), "token", PROBLEMS[:1])[0]
    assert r["exit"] == "vl_disagreement" and r["final_correct_A"] == r["iterations"][0]["correct"]


def test_l3_initially_verified_never_resamples():
    class Right(L3Engine):
        def generate(self, prompt_ids, params):
            outs = super().generate(prompt_ids, params)
            for o, ids in zip(outs, prompt_ids):
                if not self.tok.decode(ids).startswith("<chat>"):
                    o.update(text="so \\boxed{42}", ids=self.tok.encode("so \\boxed{42}"))
            return outs
    eng = Right()
    r = R.run_method(eng, _cfg(autonomy_level=3, fallback="attempt", response_length=5000), "token", PROBLEMS[:1])[0]
    assert r["exit"] == "verified" and len([i for i in r["iterations"] if "gen_status" in i]) == 1
    s = R.summarize("token", [r])
    assert s["exits"] == {"verified": 1} and s["final_accuracy_A"] == 1.0


def test_parse_verdict_latex_wrapped():
    assert P.parse_verdict("\\boxed{\\text{NO}}") is False
    assert P.parse_verdict("\\boxed{\\textbf{YES}}") is True
    assert P.parse_verdict("\\boxed{ $YES$ }") is True
    assert P.parse_verdict("no box") is None


# --- Self-Refine / CoVe baselines --------------------------------------------------------
class BaselineEngine:
    """Self-Refine feedback: NO (as gpt-oss would: analysis then final channel) until the solution
    boxes 42, then YES. Refine: boxes 42 on the 2nd refinement. CoVe: plans 2 questions."""
    def __init__(self):
        self.tok = FakeTok()
        self.max_model_len = 10**6
        self.n_refine = {}
        self.calls = []

    def gen_prompt_ids(self, text):
        return self.tok.encode("<gen>" + text)

    def chat_ids(self, text):
        return self.tok.encode("<chat>" + text)

    def generate(self, prompt_ids, params):
        outs = []
        for ids, prm in zip(prompt_ids, params):
            text = self.tok.decode(ids)
            self.calls.append((text[:6], prm.max_tokens))
            if "Provide feedback on this solution" in text:
                sol = text.split("Solution to review:")[1]
                v = "YES, correct." if "\\boxed{42}" in sol else "NO. The step is wrong."
                out = "<|channel|>analysis<|message|>thinking no yes<|end|><|start|>assistant<|channel|>final<|message|>" + v + "<|return|>"
            elif "Corrected Solution:" in text:
                pid = text.split("Problem:")[1].split("\n")[0].strip()
                self.n_refine[pid] = self.n_refine.get(pid, 0) + 1
                out = "fix \\boxed{42}" if self.n_refine[pid] >= 2 else "retry \\boxed{8}"
            elif "Verification Questions:" in text:
                out = "1. Is step one right here?\n2. Is the sum computed right?"
            elif text.startswith("<chat>Answer the following question"):
                out = "Yes."
            elif "Final Solution:" in text:
                out = "final \\boxed{42}"
            else:
                raise AssertionError(text[:80])
            outs.append({"ids": self.tok.encode(out), "text": out, "finish_reason": "stop", "stop_reason": None})
        return outs


def _bcfg(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(**{"budget": 16384, "temp": 0.5, "top_p": 0.9, "top_k": 50, "max_iterations": 10, **kw})


def test_self_refine_up_to_n_stops_on_yes():
    from thought_ics.budget_study import baselines as B
    eng = BaselineEngine()
    probs = [{"problem": "pa", "answer": "42", "unique_id": "a"}, {"problem": "pb", "answer": "42", "unique_id": "b"}]
    recs = B.run_self_refine(eng, _bcfg(), probs, ["x \\boxed{7}", "y \\boxed{42}"])
    a, b = recs
    assert b["stopped_early"] and len(b["iterations"]) == 1 and b["final_correct"]   # YES on the first attempt
    assert a["stopped_early"] and [it["answer"] for it in a["iterations"]] == ["7", "8", "42"] and a["final_correct"]
    assert a["iterations"][0]["feedback"]["text"] == "NO. The step is wrong."          # final channel only
    assert all(t == 16384 for _, t in eng.calls)


def test_self_refine_cap():
    from thought_ics.budget_study import baselines as B
    eng = BaselineEngine()
    eng.n_refine = {"pa": -100}  # never reaches 42
    recs = B.run_self_refine(eng, _bcfg(max_iterations=3), [{"problem": "pa", "answer": "42", "unique_id": "a"}], ["x \\boxed{7}"])
    assert len(recs[0]["iterations"]) == 4 and not recs[0]["stopped_early"] and not recs[0]["final_correct"]


def test_cove_single_pass():
    from thought_ics.budget_study import baselines as B
    eng = BaselineEngine()
    recs = B.run_cove(eng, _bcfg(), [{"problem": "pa", "answer": "42", "unique_id": "a"}], ["x \\boxed{7}"])
    r = recs[0]
    assert len(r["verifications"]) == 2 and len(r["iterations"]) == 2 and r["final_correct"]
    # factored: verification questions are answered without the baseline solution in context
    assert sum(1 for c, _ in eng.calls if c == "<chat>") == 3


def test_feedback_verdict_markdown():
    from thought_ics.budget_study.baselines import feedback_verdict as F
    assert F("**Answer: NO**\n\nThe solution contains several fundamental mistakes that invalidate the correct solution")[0] is False
    assert F("**YES** – the final answer \\(m+n=26\\) is correct.")[0] is True
    assert F("**1.** YES – the solution is correct.")[0] is True
    assert F("### NO\nwrong")[0] is False
    assert F("NO\n\n**Why the solution is incorrect**")[0] is False
    assert F("Yes, correct.")[0] is True


def test_fit_cuts_only_looping_eval_text():
    from thought_ics.budget_study import baselines as B
    eng = BaselineEngine()
    eng.max_model_len = B.MIN_OUTPUT_ROOM + 3000
    build = lambda ps: "SOLUTION" * 100 + "|" + "|".join(ps)
    # fits: unchanged
    p, c = B.fit(eng, build, ["short", "also short"])
    assert c == 0 and p == build(["short", "also short"])
    # one looping answer: only it is cut, solution intact, prompt fits
    p, c = B.fit(eng, build, ["ok answer", "loop " * 2000])
    assert c > 0 and p.startswith("SOLUTION" * 100 + "|ok answer|loop")
    assert len(eng.gen_prompt_ids(p)) <= eng.max_model_len - B.MIN_OUTPUT_ROOM


def test_cove_overflow_does_not_crash():
    from thought_ics.budget_study import baselines as B
    eng = BaselineEngine()
    eng.max_model_len = B.MIN_OUTPUT_ROOM + 2000
    orig = eng.generate
    def gen(ids, params):
        outs = orig(ids, params)
        for o, i in zip(outs, ids):
            if eng.tok.decode(i).startswith("<chat>Answer the following question"):
                o["text"] = "loop " * 3000
                o["finish_reason"] = "length"
        return outs
    eng.generate = gen
    r = B.run_cove(eng, _bcfg(), [{"problem": "pa", "answer": "42", "unique_id": "a"}], ["x \\boxed{7}"])[0]
    assert r["tokens_cut_to_fit"] > 0 and r["final_correct"]


class CritiqueEngine(L3Engine):
    """Verifier NO comes with a gpt-oss style critique (analysis + final message)."""
    def generate(self, prompt_ids, params):
        outs = super().generate(prompt_ids, params)
        for o, ids in zip(outs, prompt_ids):
            if self.tok.decode(ids).startswith("<chat>You are reviewing") and "NO" in o["text"]:
                o["text"] = ("<|channel|>analysis<|message|>hidden scratch<|end|><|start|>assistant"
                             "<|channel|>final<|message|>CRITIQUE: step 2 is wrong. \\boxed{NO}<|return|>")
                o["ids"] = self.tok.encode(o["text"])
        return outs


def test_carry_feedback_reaches_localizer_and_resample(monkeypatch):
    _patch_resample_marker(monkeypatch)
    eng = CritiqueEngine()
    seen = []
    orig = eng.generate
    eng.generate = lambda ids, prm: (seen.extend(eng.tok.decode(i) for i in ids), orig(ids, prm))[1]
    for method in ("thought", "token"):
        seen.clear()
        r = R.run_method(eng, _cfg(autonomy_level=3, fallback="attempt", carry_feedback=True, response_length=5000),
                         method, PROBLEMS[:1])[0]
        loc = r["iterations"][1]["localization"]
        assert loc["critique"] == "CRITIQUE: step 2 is wrong. \\boxed{NO}"        # final message only
        assert "Your check of it:\nCRITIQUE: step 2 is wrong." in loc["loc_prompt"]
        assert "hidden scratch" not in loc["loc_prompt"]
        resample = [t for t in seen if not t.startswith("<chat>") and "Feedback on a previous attempt" in t]
        assert resample and "CRITIQUE: step 2 is wrong." in resample[0]


def test_no_carry_feedback_unchanged():
    eng = CritiqueEngine()
    r = R.run_method(eng, _cfg(autonomy_level=3, fallback="attempt", response_length=5000), "token", PROBLEMS[:1])[0]
    loc = r["iterations"][1]["localization"]
    assert "critique" not in loc and "Your check of it" not in loc["loc_prompt"]


def test_fit_critique_cuts_tail():
    eng = FakeEngine()
    cfg = _cfg()
    c, cut = R.fit_critique(eng, cfg, "x" * 1000, [(lambda c: len(c) + 100, 600)])
    assert cut > 0 and len(c) + 100 <= 600 and c == "x" * len(c)
    c, cut = R.fit_critique(eng, cfg, "short", [(lambda c: len(c), 600)])
    assert cut == 0 and c == "short"
