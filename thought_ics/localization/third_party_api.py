#!/usr/bin/env python3
"""
Third-party API Module

Provides:
1. Error localization using external APIs (e.g., OpenAI GPT) instead of
   the model being evaluated. Used when --3p-localize flag is set in L2 experiments.
2. ThirdPartyModelManager: A drop-in replacement for BaseModelManager that routes
   all inference to OpenAI API. Used when --3p flag is set.
"""

import logging
import re
import time
from typing import List, Optional, Tuple, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


def call_openai_api(
    prompt: str,
    api_key: str,
    model: str = "gpt-5",
    max_retries: int = 3
) -> Tuple[str, int]:
    """
    Call OpenAI API with the error localization prompt.

    Args:
        prompt: The error localization prompt
        api_key: OpenAI API key
        model: Model to use (default: gpt-5)
        max_retries: Maximum number of retry attempts

    Returns:
        (response_text, tokens_used)

    Raises:
        RuntimeError: If OpenAI library not installed or API call fails
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI library not installed. Install with: pip install openai")

    client = OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            logger.info(f"3P API call successful ({tokens_used} tokens used)")
            return response_text, tokens_used

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"3P API error (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"3P API failed after {max_retries} attempts: {e}")
                raise RuntimeError(f"Failed to call 3P API: {e}")

    return "", 0


def call_openai_api_generate(
    prompt: str,
    api_key: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 150,
    top_p: float = 0.9,
    stop: Optional[List[str]] = None,
    max_retries: int = 10
) -> str:
    """
    Call OpenAI API for text generation with full parameter control.

    This function supports stop sequences for ToT-style generation,
    matching the behavior of vLLM's SamplingParams.

    Args:
        prompt: The generation prompt
        api_key: OpenAI API key
        model: Model to use (default: gpt-4o)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 150)
        top_p: Nucleus sampling parameter (default: 0.9)
        stop: List of stop sequences (up to 4 supported by OpenAI)
        max_retries: Maximum number of retry attempts

    Returns:
        Generated text (response content)

    Raises:
        RuntimeError: If OpenAI library not installed or API call fails
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI library not installed. Install with: pip install openai")

    client = OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            # Build API call parameters
            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            # Add stop sequences if provided (OpenAI supports up to 4)
            if stop:
                api_params["stop"] = stop[:4]  # Limit to 4 sequences

            response = client.chat.completions.create(**api_params)

            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            logger.debug(f"3P generation successful ({tokens_used} tokens)")
            return response_text

        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 200ms, 400ms, 800ms, 1.6s, 3.2s, ...
                wait_time = 0.2 * (2 ** attempt)
                logger.warning(f"3P API generation error (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"3P API generation failed after {max_retries} attempts: {e}")
                raise RuntimeError(f"Failed to call 3P API for generation: {e}")

    return ""


class ThirdPartyModelManager:
    """
    Drop-in replacement for BaseModelManager that routes all inference to OpenAI API.

    Implements the same generate() interface for compatibility with ToTAgent,
    iterative_self_correction, and other components that expect a model manager.

    When using this manager, no local GPU resources are used - all inference
    goes through the OpenAI API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
    ):
        """
        Initialize the third-party model manager.

        Args:
            api_key: OpenAI API key
            model: Model to use for generation (default: gpt-4o)
        """
        if api_key is None:
            raise ValueError("API key is required for ThirdPartyModelManager")

        self.api_key = api_key
        self.model = model
        self._loaded = True  # Always "loaded" since no GPU model to load

        # Statistics tracking (matches BaseModelManager interface)
        self.total_generations = 0
        self.total_tokens_generated = 0

        logger.info(f"ThirdPartyModelManager initialized with model: {model}")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,  # Ignored - OpenAI doesn't support top_k
        n: int = 1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text using OpenAI API. Compatible with BaseModelManager interface.

        Args:
            prompts: List of prompts to generate from
            max_tokens: Maximum tokens per generation (default: 512)
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling parameter (default: 0.9)
            top_k: Top-k sampling (IGNORED - not supported by OpenAI)
            n: Number of completions per prompt (default: 1)
            stop: List of stop sequences (default: None)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            List of generated texts. When n > 1, returns n texts per prompt
            in flattened order: [prompt1_gen1, prompt1_gen2, ..., prompt2_gen1, ...]
        """
        if top_k != 50 and top_k != -1:
            logger.debug(f"top_k={top_k} ignored (not supported by OpenAI API)")

        results = []

        for prompt in prompts:
            for _ in range(n):
                try:
                    response_text = call_openai_api_generate(
                        prompt=prompt,
                        api_key=self.api_key,
                        model=self.model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stop=stop
                    )
                    results.append(response_text)
                    self.total_generations += 1
                except Exception as e:
                    logger.error(f"Generation failed for prompt: {e}")
                    results.append("ERROR: Generation failed")

        return results

    def load_base_model(self):
        """No-op for 3P - no model to load."""
        logger.info("ThirdPartyModelManager: No local model to load (using API)")

    def unload_base_model(self):
        """No-op for 3P - no model to unload."""
        logger.info("ThirdPartyModelManager: No local model to unload (using API)")

    def get_stats(self) -> dict:
        """Get generation statistics."""
        return {
            "total_generations": self.total_generations,
            "model": self.model,
            "backend": "openai_api"
        }


def extract_step_number(text: str) -> Optional[int]:
    """
    Extract step number from model response.

    Looks for patterns like:
    - \\boxed{5}
    - \boxed{5}

    Returns None if not found.
    """
    # Look for \boxed{number} or \\boxed{number}
    pattern = r'\\*boxed\{(\d+)\}'
    matches = re.findall(pattern, text)

    if matches:
        # Take the last occurrence (model's final decision)
        return int(matches[-1])

    # Fallback: search for any number in the response
    # (matches original behavior in iterative_self_correction.py)
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        logger.warning(f"Could not find \\boxed{{number}}, using fallback number: {numbers[-1]}")
        return int(numbers[-1])

    return None


def extract_yes_no(text: str) -> Optional[str]:
    """
    Extract YES/NO from incremental mode response.

    Returns 'YES', 'NO', or None
    """
    # Look for \boxed{YES} or \boxed{NO}
    pattern = r'\\*boxed\{(YES|NO)\}'
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        return matches[-1].upper()

    return None


def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from \\boxed{} format.

    This matches the implementation in baseline_cot_eval.py.
    Handles nested braces correctly.

    Returns:
        The content inside the last \\boxed{} in the text,
        or "NO ANSWER" if not found.
    """
    if not text:
        return "NO ANSWER"

    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return "NO ANSWER"

    start_pos = matches[-1].end()
    brace_count = 1
    i = start_pos
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return text[start_pos:i-1].strip()

    return "NO ANSWER"


def construct_l2_batch_prompt(problem: str, chain: List[str]) -> str:
    """
    Construct L2 batch mode error localization prompt.

    This matches the prompt from iterative_self_correction.py lines 269-279.

    Args:
        problem: The problem statement
        chain: List of reasoning steps

    Returns:
        Formatted prompt string
    """
    # Build chain text
    chain_text = ""
    for i, step in enumerate(chain, 1):
        chain_text += f"\nStep {i}: {step}"

    prompt = f"""Problem: {problem}

Current reasoning chain (WRONG - got incorrect answer):
{chain_text}

Your answer is incorrect. Analyze the reasoning chain step by step to identify where the error occurred. Which step number (1 to {len(chain)}) contains the first critical error (logical flaw, arithmetic error, or incorrect assumption)?

Provide your reasoning, then conclude with the step number in the format: \\boxed{{step_number}}
"""

    return prompt


def construct_l2_incremental_prompt(
    problem: str,
    chain: List[str],
    step_idx: int
) -> str:
    """
    Construct L2 incremental mode error verification prompt for a specific step.

    This matches the prompt from iterative_self_correction.py lines 123-140.

    Args:
        problem: The problem statement
        chain: List of reasoning steps
        step_idx: Index of step to verify (1-indexed)

    Returns:
        Formatted prompt string
    """
    # Build context (previous steps)
    context_text = ""
    if step_idx > 1:
        context_text = "\n\nPrevious steps (already verified):"
        for i in range(step_idx - 1):
            context_text += f"\nStep {i + 1}: {chain[i]}"

    current_step_text = chain[step_idx - 1]

    prompt = f"""Problem: {problem}
{context_text}

Current step to verify:
Step {step_idx}: {current_step_text}

You are verifying a reasoning chain that led to an incorrect answer.

Question: Is Step {step_idx} logically correct and mathematically accurate given the problem{' and previous steps' if step_idx > 1 else ''}?

Analyze this specific step carefully. Then respond:
- \\boxed{{YES}} if Step {step_idx} is correct
- \\boxed{{NO}} if Step {step_idx} contains an error (logical flaw, arithmetic error, or incorrect assumption)

Provide your reasoning first, then your conclusion.
"""

    return prompt


def call_3p_error_localization_batch(
    problem: str,
    chain: List[str],
    ground_truth: str,
    api_key: str,
    model: str = "gpt-5"
) -> Tuple[int, str]:
    """
    Use 3rd-party API for batch mode error localization (L2).

    Args:
        problem: Problem statement
        chain: Reasoning chain (list of steps)
        ground_truth: Ground truth answer (not used in prompt, for logging only)
        api_key: API key for 3rd-party service
        model: Model to use (default: gpt-5)

    Returns:
        (step_number, reasoning_text)
        step_number: Which step has the error (1-indexed)
        reasoning_text: Full reasoning from the model
    """
    logger.info(f"Using 3P API ({model}) for batch error localization...")

    # Construct L2 prompt
    prompt = construct_l2_batch_prompt(problem, chain)

    # Call 3P API
    response, tokens = call_openai_api(prompt, api_key, model)

    # Extract step number
    step_num = extract_step_number(response)

    if step_num is None:
        # Fallback: return middle of chain (matches original behavior)
        step_num = max(1, len(chain) // 2)
        logger.warning(f"Could not parse step number from 3P response, using fallback: step {step_num}")

    logger.info(f"3P API identified error at step {step_num}")

    return step_num, response


def call_3p_error_localization_incremental(
    problem: str,
    chain: List[str],
    ground_truth: str,
    api_key: str,
    model: str = "gpt-5"
) -> Tuple[int, str]:
    """
    Use 3rd-party API for incremental mode error localization (L2).

    Verifies each step sequentially until an error is found.

    Args:
        problem: Problem statement
        chain: Reasoning chain (list of steps)
        ground_truth: Ground truth answer (not used in prompt, for logging only)
        api_key: API key for 3rd-party service
        model: Model to use (default: gpt-5)

    Returns:
        (step_number, reasoning_text)
        step_number: First step with error (1-indexed), or 0 if all correct
        reasoning_text: Combined reasoning from all verifications
    """
    logger.info(f"Using 3P API ({model}) for incremental error localization...")

    all_reasoning = []

    for step_idx in range(1, len(chain) + 1):
        logger.info(f"Verifying step {step_idx}/{len(chain)} with 3P API...")

        # Construct L2 incremental prompt for this step
        prompt = construct_l2_incremental_prompt(problem, chain, step_idx)

        # Call 3P API
        response, tokens = call_openai_api(prompt, api_key, model)
        all_reasoning.append(f"Step {step_idx}: {response}")

        # Extract YES/NO
        decision = extract_yes_no(response)

        if decision == "NO":
            logger.info(f"3P API found error at step {step_idx}")
            combined_reasoning = "\n\n".join(all_reasoning)
            return step_idx, combined_reasoning
        elif decision == "YES":
            logger.info(f"Step {step_idx} verified as correct")
            continue
        else:
            # Could not parse - assume error for safety
            logger.warning(f"Could not parse YES/NO from 3P response for step {step_idx}, assuming error")
            combined_reasoning = "\n\n".join(all_reasoning)
            return step_idx, combined_reasoning

    # All steps passed
    logger.info("All steps verified as correct by 3P API")
    combined_reasoning = "\n\n".join(all_reasoning)
    return 0, combined_reasoning


def call_3p_error_localization(
    problem: str,
    chain: List[str],
    ground_truth: str,
    autonomy_level: int,
    method: str = "batch",
    api_key: Optional[str] = None,
    model: str = "gpt-5"
) -> Tuple[int, str]:
    """
    Main entry point for 3rd-party error localization.

    Args:
        problem: Problem statement
        chain: Reasoning chain (list of steps)
        ground_truth: Ground truth answer
        autonomy_level: Autonomy level (should be 2 for L2)
        method: 'batch' or 'incremental'
        api_key: API key for 3rd-party service
        model: Model to use (default: gpt-5)

    Returns:
        (step_number, reasoning_text)

    Raises:
        ValueError: If api_key not provided or autonomy_level != 2
    """
    if api_key is None:
        raise ValueError("3P API key is required when using 3rd-party localization")

    if autonomy_level != 2:
        logger.warning(f"3P localization called with autonomy_level={autonomy_level}, expected 2")

    if method == "incremental":
        return call_3p_error_localization_incremental(
            problem, chain, ground_truth, api_key, model
        )
    else:  # default: batch
        return call_3p_error_localization_batch(
            problem, chain, ground_truth, api_key, model
        )


def call_3p_error_localization_cot_quote(
    problem: str,
    solution: str,
    ground_truth: str,
    api_key: str,
    model: str = "gpt-5"
) -> Tuple[Optional[str], str]:
    """
    Use 3rd-party API for token-level error localization in CoT (shared prefix mode).

    This matches the behavior of identify_error_location_shared_prefix() in baseline_cot_eval.py,
    but uses a 3rd-party API instead of the evaluated model.

    Args:
        problem: Problem statement
        solution: Full solution text (continuous string, not discrete steps)
        ground_truth: Ground truth answer (not used in L2 prompt)
        api_key: API key for 3rd-party service
        model: Model to use (default: gpt-5)

    Returns:
        (quoted_error_text, reasoning)
        quoted_error_text: The exact text where the error begins, or None if no error found
        reasoning: Full reasoning from the model
    """
    logger.info(f"Using 3P API ({model}) for CoT error localization (quote mode)...")

    # Construct L2 prompt (matches baseline_cot_eval.py lines 110-123)
    prompt = f"""Problem: {problem}

Current solution (WRONG - got incorrect answer):
{solution}

Your answer is incorrect. Analyze the solution step by step to identify where the error occurred. Quote the EXACT text (word-for-word) where the first critical error (logical flaw, arithmetic error, or incorrect assumption) begins. This should be a continuous excerpt from your solution above.

Provide your reasoning, then conclude with the exact quote in the format:
\\boxed{{ERROR_QUOTE: "exact text from solution where error occurs"}}

If you cannot find the error, respond with: \\boxed{{NO_ERROR}}
"""

    # Call 3P API
    response, tokens = call_openai_api(prompt, api_key, model)

    # Extract boxed answer
    boxed = extract_boxed_answer(response)

    if boxed == "CORRECT" or boxed == "NO_ERROR":
        logger.info("3P API found no errors")
        return None, response

    # Try to extract quoted text
    # Format: ERROR_QUOTE: "quoted text"
    if "ERROR_QUOTE:" in boxed:
        quote_match = re.search(r'ERROR_QUOTE:\s*"([^"]+)"', boxed)
        if quote_match:
            quoted_text = quote_match.group(1)
            logger.info(f"3P API extracted error quote: {quoted_text[:100]}...")
            return quoted_text, response

    # Fallback: try to find any quoted text in the response (20+ chars)
    quote_matches = re.findall(r'"([^"]{20,})"', response)
    if quote_matches:
        quoted_text = quote_matches[0]
        logger.info(f"3P API found quoted text (fallback): {quoted_text[:100]}...")
        return quoted_text, response

    logger.warning("Could not extract error quote from 3P API response")
    return None, response
