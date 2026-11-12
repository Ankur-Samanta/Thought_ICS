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
                token_prompt="token", token_init_from=None)
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
