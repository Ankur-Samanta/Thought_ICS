"""CPU tests for thought_ics with a scripted fake engine (no vLLM / GPU needed)."""

import types

from thought_ics import protocol as P
from thought_ics import ics as R


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



def _token_init(problems, chain=("good part wrong part so \\boxed{7}",)):
    """Thought-ICS records that Token-ICS starts from (the same job's first attempts)."""
    return [{"problem_id": p["unique_id"], "empty_retries": 0,
             "iterations": [{"iteration": 0, "chain": list(chain), "gen_status": "answer"}]} for p in problems]

def _cfg(**kw):
    base = dict(max_thoughts=P.MAX_THOUGHTS, max_tokens_per_thought=P.MAX_TOKENS_PER_THOUGHT,
                response_length=10**5, localization_max_tokens=P.LOCALIZATION_MAX_TOKENS,
                localization_temp=P.LOCALIZATION_TEMP, generation_temp=P.GENERATION_TEMP,
                top_p=P.TOP_P, top_k=P.TOP_K, max_ics_iterations=3,
                max_empty_retries=P.MAX_EMPTY_RETRIES, verify="oracle", verifier=None, feedback=False, init_from=None,
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
    recs = R.run_method(eng, _cfg(response_length=5000), "token", PROBLEMS[:1],
                     shared_init=_token_init(PROBLEMS[:1]))
    r = recs[0]
    assert not r["iterations"][0]["correct"]
    loc = r["iterations"][1]["localization"]
    assert loc["parse_status"] == "boxed" and loc["error_quote"] == "wrong part" and loc["quote_found"]
    assert r["final_correct"]
    # generation budget = response_length minus prefix; localization budget = LOCALIZATION_MAX_TOKENS
    gen_calls = [m for chat, m, s in eng.calls if not s and not chat]
    assert gen_calls == [5000 - len("good part")]   # resample: budget minus the kept prefix
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
    assert P.parse_error_step("checked it\n\n4", 5) == (4, "bare")     # a bare step number as the last line
    assert P.parse_error_step("**0**", 5) == (0, "bare")
    assert P.parse_error_step("the answer is 12", 5) == (1, "default")       # not a step number of this chain
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


def test_token_shared_initial_cache():
    """Token-ICS starts from the joined Thought-ICS chain and resamples with the same prompt."""
    chain = ["good part ", "wrong part so \\boxed{7}"]
    rec = {"problem_id": PROBLEMS[0]["unique_id"], "empty_retries": 0,
           "iterations": [{"iteration": 0, "chain": chain, "gen_status": "answer"}]}
    eng = FakeEngine()
    cfg = _cfg(response_length=5000)
    r = R.run_method(eng, cfg, "token", PROBLEMS[:1], shared_init=[rec])[0]
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
    """Unusable localization -> no resample; the next iteration re-localizes."""
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
    r = R.run_method(eng, _cfg(response_length=5000, max_ics_iterations=3),
                     "token", PROBLEMS[:1],
                     shared_init=_token_init(PROBLEMS[:1]))[0]
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
    r = R.run_method(eng, _cfg(response_length=5000, max_ics_iterations=2),
                     "token", PROBLEMS[:1],
                     shared_init=_token_init(PROBLEMS[:1]))[0]
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


class SelfVerifyEngine(FakeEngine):
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


def test_self_verified_correction(monkeypatch):
    _patch_resample_marker(monkeypatch)
    eng = SelfVerifyEngine()
    r = R.run_method(eng, _cfg(verify="self", verifier="self-refine"), "thought", PROBLEMS[:1])[0]
    its = r["iterations"]
    assert not its[0]["correct"]
    assert its[1]["localization"]["verification"]["verdict"] is False and its[1]["correct"]
    assert its[2].get("stopped") == "verified" and r["exit"] == "verified"
    assert r["final_correct"]


def test_self_verified_vl_disagreement():
    eng = SelfVerifyEngine()
    eng.loc_no_error = True
    r = R.run_method(eng, _cfg(verify="self", verifier="self-refine", response_length=5000), "token", PROBLEMS[:1],
                     shared_init=_token_init(PROBLEMS[:1]))[0]
    assert r["exit"] == "vl_disagreement" and r["final_correct"] == r["iterations"][0]["correct"]


def test_self_verified_initially_verified_never_resamples():
    class Right(SelfVerifyEngine):
        def generate(self, prompt_ids, params):
            outs = super().generate(prompt_ids, params)
            for o, ids in zip(outs, prompt_ids):
                if not self.tok.decode(ids).startswith("<chat>"):
                    o.update(text="so \\boxed{42}", ids=self.tok.encode("so \\boxed{42}"))
            return outs
    eng = Right()
    r = R.run_method(eng, _cfg(verify="self", verifier="self-refine", response_length=5000), "token", PROBLEMS[:1],
                     shared_init=_token_init(PROBLEMS[:1], chain=("so \\boxed{42}",)))[0]
    assert r["exit"] == "verified" and len([i for i in r["iterations"] if "gen_status" in i]) == 1
    s = R.summarize("token", [r])
    assert s["exits"] == {"verified": 1} and s["final_accuracy"] == 1.0


def test_parse_verdict_latex_wrapped():
    assert P.parse_verdict("\\boxed{\\text{NO}}") is False
    assert P.parse_verdict("\\boxed{\\textbf{YES}}") is True
    assert P.parse_verdict("\\boxed{ $YES$ }") is True
    assert P.parse_verdict("no box") is None
    assert P.parse_verdict("so the answer is correct.\nYES") is True        # off-format final verdict
    assert P.parse_verdict("boxed{NO}<|eot_id|>") is False
    assert P.parse_verdict("yes, I think so") is None                       # only an upper-case YES / NO counts


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
            self.prompts = getattr(self, "prompts", []) + [text]
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
                fa = getattr(self, "final_answers", None)
                out = f"final \\boxed{{{next(fa) if fa else 42}}}"
            else:
                raise AssertionError(text[:80])
            outs.append({"ids": self.tok.encode(out), "text": out, "finish_reason": "stop", "stop_reason": None})
        return outs


def _bcfg(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(**{"budget": 16384, "temp": 0.5, "top_p": 0.9, "top_k": 50, "max_iterations": 10, "dataset": "aime", **kw})


def test_self_refine_up_to_n_stops_on_yes():
    from thought_ics import baselines as B
    eng = BaselineEngine()
    probs = [{"problem": "pa", "answer": "42", "unique_id": "a"}, {"problem": "pb", "answer": "42", "unique_id": "b"}]
    recs = B.run_self_refine(eng, _bcfg(), probs, ["x \\boxed{7}", "y \\boxed{42}"])
    a, b = recs
    assert b["stopped_early"] and len(b["iterations"]) == 1 and b["final_correct"]   # YES on the first attempt
    assert a["stopped_early"] and [it["answer"] for it in a["iterations"]] == ["7", "8", "42"] and a["final_correct"]
    assert a["iterations"][0]["feedback"]["text"] == "NO. The step is wrong."          # final channel only
    assert all(t == 16384 for _, t in eng.calls)


def test_self_refine_cap():
    from thought_ics import baselines as B
    eng = BaselineEngine()
    eng.n_refine = {"pa": -100}  # never reaches 42
    recs = B.run_self_refine(eng, _bcfg(max_iterations=3), [{"problem": "pa", "answer": "42", "unique_id": "a"}], ["x \\boxed{7}"])
    assert len(recs[0]["iterations"]) == 4 and not recs[0]["stopped_early"] and not recs[0]["final_correct"]


def test_cove_repeats_until_answer_unchanged():
    from thought_ics import baselines as B
    eng = BaselineEngine()
    recs = B.run_cove(eng, _bcfg(), [{"problem": "pa", "answer": "42", "unique_id": "a"}], ["x \\boxed{7}"])
    r = recs[0]
    # round 1 revises 7 -> 42, round 2 leaves 42 unchanged: stop
    assert [it["answer"] for it in r["iterations"]] == ["7", "42", "42"] and r["final_correct"]
    assert len(r["rounds"]) == 2 and all(len(rd["verifications"]) == 2 for rd in r["rounds"])
    # factored: verification questions are answered without the solution in context
    asks = [t for t in eng.prompts if t.startswith("<chat>Answer the following question")]
    assert len(asks) == 4 and not any("boxed" in t for t in asks)


def test_cove_iteration_cap():
    from thought_ics import baselines as B
    eng = BaselineEngine()
    eng.final_answers = iter(range(100))     # every revision changes the answer
    r = B.run_cove(eng, _bcfg(max_iterations=3), [{"problem": "pa", "answer": "42", "unique_id": "a"}], ["x \\boxed{7}"])[0]
    assert len(r["rounds"]) == 3 and len(r["iterations"]) == 4


def test_feedback_verdict_markdown():
    from thought_ics.baselines import feedback_verdict as F
    assert F("**Answer: NO**\n\nThe solution contains several fundamental mistakes that invalidate the correct solution")[0] is False
    assert F("**YES** – the final answer \\(m+n=26\\) is correct.")[0] is True
    assert F("**1.** YES – the solution is correct.")[0] is True
    assert F("### NO\nwrong")[0] is False
    assert F("NO\n\n**Why the solution is incorrect**")[0] is False
    assert F("Yes, correct.")[0] is True


def test_fit_cuts_only_looping_eval_text():
    from thought_ics import baselines as B
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
    from thought_ics import baselines as B
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
    finals = [t for t in eng.prompts if "Final Solution:" in t]
    assert r["final_correct"] and all(len(t) <= eng.max_model_len - B.MIN_OUTPUT_ROOM for t in finals)


class CritiqueEngine(SelfVerifyEngine):
    """Verifier NO comes with a gpt-oss style critique (analysis + final message)."""
    def generate(self, prompt_ids, params):
        outs = super().generate(prompt_ids, params)
        for o, ids in zip(outs, prompt_ids):
            if self.tok.decode(ids).startswith("<chat>You are reviewing") and "NO" in o["text"]:
                o["text"] = ("<|channel|>analysis<|message|>hidden scratch<|end|><|start|>assistant"
                             "<|channel|>final<|message|>CRITIQUE: step 2 is wrong. \\boxed{NO}<|return|>")
                o["ids"] = self.tok.encode(o["text"])
        return outs


def test_carry_feedback(monkeypatch):
    _patch_resample_marker(monkeypatch)
    for method in ("thought", "token"):
        eng = CritiqueEngine()
        seen = []
        orig = eng.generate
        eng.generate = lambda ids, prm, orig=orig, seen=seen, eng=eng: (seen.extend(eng.tok.decode(i) for i in ids), orig(ids, prm))[1]
        r = R.run_method(eng, _cfg(verify="self", verifier="self-refine", feedback=True, response_length=5000), method,
                         PROBLEMS[:1], shared_init=_token_init(PROBLEMS[:1]) if method == "token" else None)[0]
        loc = r["iterations"][1]["localization"]
        assert loc["critique"] == "hidden scratch\n\nCRITIQUE: step 2 is wrong. \\boxed{NO}"  # reasoning + final
        assert "Your check of it:\nhidden scratch\n\nCRITIQUE: step 2 is wrong." in loc["loc_prompt"]
        assert "<|channel|>" not in loc["loc_prompt"]
        fb = [t for t in seen if not t.startswith("<chat>") and "Feedback on a previous attempt" in t]
        assert fb and "CRITIQUE: step 2 is wrong." in fb[0]
        assert "Where the previous attempt first went wrong:\n" + P.unbox(loc["resample_localization"]) in fb[0]
        assert P.FEEDBACK_GEN_CLOSE in fb[0] and "\\boxed{NO}" not in fb[0].split("Feedback on a previous attempt")[1].split("Taking this feedback")[0]
        assert loc["resample_localization"] == R.response_text(loc["response"]).strip()


class CoveVerifyEngine(SelfVerifyEngine):
    """CoVe-style verifier: plans two questions; answers them; the judge says YES iff the solution
    boxes 42. With off_format, the judge gives no verdict and the follow-up completes it."""
    off_format = False

    def generate(self, prompt_ids, params):
        outs, rest = [], []
        for ids, prm in zip(prompt_ids, params):
            text = self.tok.decode(ids)
            if text.endswith(P.VERDICT_FOLLOWUP):
                out = "YES}" if "looks fine" in text else "NO}"
            elif text.startswith("<chat>You are checking") and "Independent verification:" not in text:
                out = "Q1: What is 6*7?\nQ2: Is step 2 justified?"
            elif text.startswith("<chat>Answer the following question about a problem"):
                assert "Solution to check" not in text and "step 1 bad" not in text   # problem only
                out = "It is 42."
            elif text.startswith("<chat>You are checking"):
                ok = "Final answer: 42" in text
                out = ("looks fine" if ok else "looks wrong") if self.off_format else ("\\boxed{YES}" if ok else "\\boxed{NO}")
            else:
                outs.append(None); rest.append((ids, prm)); continue
            outs.append({"ids": self.tok.encode(out), "text": out, "finish_reason": "stop", "stop_reason": None})
        sub = iter(super().generate([r[0] for r in rest], [r[1] for r in rest]) if rest else [])
        return [o if o is not None else next(sub) for o in outs]


def test_cove_style_verification(monkeypatch):
    _patch_resample_marker(monkeypatch)
    for off_format in (False, True):
        eng = CoveVerifyEngine()
        eng.off_format = off_format
        r = R.run_method(eng, _cfg(verify="self", verifier="cove"), "thought", PROBLEMS[:1])[0]
        its = r["iterations"]
        v = its[1]["localization"]["verification"]
        assert v["verdict"] is False and v["factored"]["questions"] == 2
        assert "Step 1: step 0 ok" in v["verify_prompt"] and "Q: What is 6*7?\nA: It is 42." in v["verify_prompt"]
        assert ("verdict_followup" in v) == off_format
        assert its[1]["correct"] and its[2]["stopped"] == "verified" and r["final_correct"]
    # with feedback, the judge's reply is the verification reasoning carried to the localizer
    r = R.run_method(CoveVerifyEngine(), _cfg(verify="self", verifier="cove", feedback=True), "thought", PROBLEMS[:1])[0]
    loc = r["iterations"][1]["localization"]
    assert loc["critique"] == R.response_text(loc["verification"]["response"]).strip()
    assert "Your check of it:\n" + loc["critique"] in loc["loc_prompt"]


def test_unparseable_verdict_counts_as_no():
    class NoVerdict(SelfVerifyEngine):
        def generate(self, prompt_ids, params):
            outs = super().generate(prompt_ids, params)
            for o, ids in zip(outs, prompt_ids):
                t = self.tok.decode(ids)
                if t.startswith("<chat>You are reviewing") or t.endswith(P.VERDICT_FOLLOWUP):
                    o.update(text="hmm", ids=self.tok.encode("hmm"))
            return outs
    eng = NoVerdict()
    r = R.run_method(eng, _cfg(verify="self", verifier="self-refine", response_length=5000), "token", PROBLEMS[:1],
                     shared_init=_token_init(PROBLEMS[:1]))[0]
    v = r["iterations"][1]["localization"]["verification"]
    assert v["verdict"] is None and v["verdict_followup"] == "hmm" and "gen_status" in r["iterations"][1]


def test_run_tags():
    assert R.MODE_TAG[("self", "cove", True)] == "thought-ics-f"
    assert len(set(R.MODE_TAG.values())) == len(R.MODE_TAG)


def test_no_carry_feedback_unchanged():
    eng = CritiqueEngine()
    r = R.run_method(eng, _cfg(verify="self", verifier="self-refine", response_length=5000), "token", PROBLEMS[:1],
                     shared_init=_token_init(PROBLEMS[:1]))[0]
    loc = r["iterations"][1]["localization"]
    assert "critique" not in loc and "Your check of it" not in loc["loc_prompt"]


def test_fit_critique_cuts_tail():
    eng = FakeEngine()
    cfg = _cfg()
    c, cut = R.fit_critique(eng, cfg, "x" * 1000, [(lambda c: len(c) + 100, 600)])
    assert cut > 0 and len(c) + 100 <= 600 and c == "x" * len(c)
    c, cut = R.fit_critique(eng, cfg, "short", [(lambda c: len(c), 600)])
    assert cut == 0 and c == "short"


def test_response_text_harmony():
    t = ("<|channel|>analysis<|message|>think A<|end|><|start|>assistant<|channel|>final<|message|>"
         "verdict \\boxed{NO}<|return|>")
    assert R.response_text(t) == "think A\n\nverdict \\boxed{NO}"
    assert R.response_text("plain \\boxed{3}<|eot_id|>") == "plain \\boxed{3}"


def test_cove_overflow_from_question_does_not_crash():
    from thought_ics import baselines as B
    eng = BaselineEngine()
    eng.max_model_len = B.MIN_OUTPUT_ROOM + 2000
    orig = eng.generate
    def gen(ids, params):
        outs = orig(ids, params)
        for o, i in zip(outs, ids):
            if "Verification Questions:" in eng.tok.decode(i):
                o["text"] = "1. " + "loop " * 3000 + "?"
        return outs
    eng.generate = gen
    B.run_cove(eng, _bcfg(), [{"problem": "pa", "answer": "42", "unique_id": "a"}], ["x \\boxed{7}"])
    finals = [t for t in eng.prompts if "Final Solution:" in t]
    assert finals and all(len(t) <= eng.max_model_len - B.MIN_OUTPUT_ROOM for t in finals)


def test_unbox():
    assert P.unbox("x \\boxed{NO} y \\boxed{\\frac{1}{2}} z") == "x NO y \\frac{1}{2} z"
    assert P.unbox("no box") == "no box"
    assert P.unbox("open \\boxed{3") == "open 3"


def test_majority_vote():
    from thought_ics.majority_vote import vote
    assert vote(["5.5", "\\frac{11}{2}", "3", "3"], "math500") == ("5.5", 2)        # equivalent forms pool; tie -> earliest
    assert vote(["A", "B", "B", "NO ANSWER"], "csqa") == ("B", 2)
    assert vote(["NO ANSWER"] * 3, "aime") == ("NO ANSWER", 0)
    assert vote(["x=5", "5", "7"], "amc23")[1] in (1, 2)


def test_self_refine_carries_full_feedback():
    from thought_ics import baselines as B
    eng = BaselineEngine()
    seen = []
    orig = eng.generate
    eng.generate = lambda ids, prm: (seen.extend(eng.tok.decode(i) for i in ids), orig(ids, prm))[1]
    B.run_self_refine(eng, _bcfg(max_iterations=1), [{"problem": "pa", "answer": "42", "unique_id": "a"}], ["x \\boxed{7}"])
    refine = [t for t in seen if "Corrected Solution:" in t]
    assert refine and "thinking no yes" in refine[0] and "NO. The step is wrong." in refine[0]   # reasoning + final
    assert "<|channel|>" not in refine[0]
