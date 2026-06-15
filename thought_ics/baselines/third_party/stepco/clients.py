"""HTTP clients for the two vLLM OpenAI-compatible servers used by StepCo.

- BaseModelClient: client for the reasoner LLM (used for CoT generation,
  rectification, and answer extraction). Calls /v1/completions for plain
  completion-style prompting (matching StepCo's prompt templates).

- MathShepherdClient: client for the Math-Shepherd PRM verifier. Uses vLLM's
  `prompt_logprobs` extension on /v1/completions to extract per-step scores.

Both clients are stateless and thread-safe.
"""

import math
import logging
from typing import List, Optional

import requests
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# =============================================================================
# Base reasoner client
# =============================================================================

class BaseModelClient:
    """vLLM HTTP client for the base reasoner model."""

    def __init__(self, base_url: str, model: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate a completion. Returns the generated text only (no echo)."""
        body = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        # vLLM accepts top_k as an extra param; -1 disables
        if top_k is not None and top_k > 0:
            body["top_k"] = top_k
        if stop:
            body["stop"] = stop
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"]


# =============================================================================
# Math-Shepherd PRM verifier client
# =============================================================================

MATH_SHEPHERD_MODEL = "peiyi9979/math-shepherd-mistral-7b-prm"
GOOD_TOKEN = "+"
BAD_TOKEN = "-"
STEP_TAG = "ки"


class MathShepherdClient:
    """vLLM HTTP client for the Math-Shepherd PRM.

    Uses vLLM's `prompt_logprobs` extension to extract per-step scores in a
    single forward pass. The protocol mirrors StepCo's verification.py:
    softmax over the model's logits for tokens '+' and '-' at the position
    immediately following each `ки` step tag.

    Implementation detail: vLLM's prompt_logprobs returns the model's
    distribution at each prompt position (given the prefix). To get the
    distribution at the position *after* each ки, we append a placeholder
    '+' token after each ки in the input. The prompt_logprobs entry at that
    placeholder position then contains exactly the model's predicted
    distribution over what should come after ки -- which is what StepCo's
    raw-logit code reads.
    """

    def __init__(
        self,
        base_url: str,
        model: str = MATH_SHEPHERD_MODEL,
        timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        # Load tokenizer locally (just the tokenizer config files, no model
        # weights) to resolve special token IDs and locate placeholder positions.
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # Resolve token IDs the same way StepCo's verification.py does.
        self.candidate_token_ids = self.tokenizer.encode(
            f"{GOOD_TOKEN} {BAD_TOKEN}"
        )[1:]  # [+, -]
        if len(self.candidate_token_ids) != 2:
            raise RuntimeError(
                f"Expected 2 candidate tokens for '+ -', got "
                f"{self.candidate_token_ids}"
            )
        self.good_token_id = self.candidate_token_ids[0]
        self.bad_token_id = self.candidate_token_ids[1]
        self.step_tag_id = self.tokenizer.encode(STEP_TAG)[-1]
        logger.info(
            f"MathShepherdClient: good={self.good_token_id} "
            f"bad={self.bad_token_id} step_tag={self.step_tag_id}"
        )

    def step_verify_score(self, input_seq: str) -> List[float]:
        """Score each step (each occurrence of ки) in `input_seq`.

        Returns a list of probabilities (one per ки), where each probability
        is softmax(+ logit, - logit)[+] at the position right after that ки.
        """
        # Append a '+' placeholder right after each " ки" so we have a token
        # at the position where we want to read the model's prediction.
        # StepCo's inputs always put a space before ки.
        modified = input_seq.replace(f" {STEP_TAG}", f" {STEP_TAG} {GOOD_TOKEN}")

        # Tokenize locally to find placeholder positions.
        token_ids = self.tokenizer.encode(modified)
        placeholder_positions = []
        for i in range(1, len(token_ids)):
            if token_ids[i - 1] == self.step_tag_id and token_ids[i] == self.good_token_id:
                placeholder_positions.append(i)

        if not placeholder_positions:
            logger.warning(
                f"No (ки, +) pairs found in tokenized input. "
                f"Input had {input_seq.count(STEP_TAG)} ки occurrences. Returning empty."
            )
            return []

        # Call vLLM with prompt_logprobs to get top-K logprobs at each prompt
        # position. max_tokens=1 and temperature=0 since we only care about
        # prompt logprobs, not generation.
        body = {
            "model": self.model,
            "prompt": modified,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 20,
        }
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        prompt_logprobs = data["choices"][0].get("prompt_logprobs")
        if prompt_logprobs is None:
            raise RuntimeError(
                "vLLM server did not return prompt_logprobs. "
                "Is the server running with the OpenAI-compatible API and a recent vLLM version?"
            )

        scores: List[float] = []
        for pos in placeholder_positions:
            if pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
                scores.append(0.0)
                continue
            entry = prompt_logprobs[pos]
            good_lp, bad_lp = self._extract_candidate_logprobs(entry)
            if good_lp is None and bad_lp is None:
                scores.append(0.0)
                continue
            if good_lp is None:
                scores.append(0.0)
                continue
            if bad_lp is None:
                scores.append(1.0)
                continue
            # Softmax over (+, -)
            m = max(good_lp, bad_lp)
            ge = math.exp(good_lp - m)
            be = math.exp(bad_lp - m)
            scores.append(ge / (ge + be))
        return scores

    def _extract_candidate_logprobs(self, entry):
        """Pull the logprobs for + and - from a prompt_logprobs entry.

        vLLM's prompt_logprobs entries are dicts keyed by token id (sometimes
        as int, sometimes as str depending on JSON serialization), with
        values that are dicts containing 'logprob' (and optionally
        'decoded_token', 'rank').
        """
        good_lp = None
        bad_lp = None
        for k, v in entry.items():
            try:
                tok_id = int(k)
            except (TypeError, ValueError):
                continue
            lp = v.get("logprob") if isinstance(v, dict) else v
            if tok_id == self.good_token_id:
                good_lp = lp
            elif tok_id == self.bad_token_id:
                bad_lp = lp
        return good_lp, bad_lp
