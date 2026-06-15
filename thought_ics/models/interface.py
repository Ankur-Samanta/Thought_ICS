"""
Unified interface for SIERA model management.
Provides simple, high-level API for common operations.
"""
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .pool import ModelPool
from .base_manager import BaseModelManager
from .config import LoRAConfig, PriorExtractionResult

logger = logging.getLogger(__name__)


class SIERAModel:
    """
    Unified interface for SIERA model management.

    This class provides a simple API for loading models, managing adapters,
    and performing common operations without dealing with pool management details.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        default_device: Optional[str] = None,
        default_quantization: Optional[str] = None
    ):
        """
        Initialize the SIERA model interface.

        Args:
            config_path: Path to model configuration file
            default_device: Default device for models ('cuda', 'cpu', etc.)
            default_quantization: Default quantization ('4bit', '8bit', None)
        """
        self.pool = ModelPool(
            config_path=config_path,
            default_device=default_device,
            default_quantization=default_quantization
        )
        self._current_model: Optional[str] = None
        self._current_adapter: Optional[str] = None

    @property
    def current_model(self) -> Optional[str]:
        """Get the currently selected model nickname."""
        return self._current_model

    @property
    def current_adapter(self) -> Optional[str]:
        """Get the currently selected adapter ID."""
        return self._current_adapter

    @property
    def available_models(self) -> List[str]:
        """Get list of all available model nicknames."""
        return self.pool.available_models

    @property
    def loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return self.pool.loaded_models

    def use_model(self, model_nickname: str, **kwargs) -> 'SIERAModel':
        """
        Set the current model to use.

        Args:
            model_nickname: Model nickname to use
            **kwargs: Additional arguments for model loading

        Returns:
            Self for chaining
        """
        self.pool.load_model(model_nickname, **kwargs)
        self._current_model = model_nickname
        self._current_adapter = None  # Reset adapter when switching models
        return self

    def load_adapter(
        self,
        adapter_id: str,
        adapter_path: Optional[str] = None,
        model_nickname: Optional[str] = None,
        **kwargs
    ) -> 'SIERAModel':
        """
        Load an adapter.

        Args:
            adapter_id: Unique identifier for the adapter
            adapter_path: Path to saved adapter (if loading existing)
            model_nickname: Model to load adapter into (uses current if None)
            **kwargs: Additional arguments for adapter loading

        Returns:
            Self for chaining
        """
        target_model = model_nickname or self._current_model
        if target_model is None:
            raise ValueError("No model selected. Use use_model() first.")

        self.pool.load_adapter(
            model_nickname=target_model,
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            **kwargs
        )
        return self

    def use_adapter(self, adapter_id: str, model_nickname: Optional[str] = None) -> 'SIERAModel':
        """
        Activate an adapter.

        Args:
            adapter_id: Adapter to activate
            model_nickname: Model to activate adapter on (uses current if None)

        Returns:
            Self for chaining
        """
        target_model = model_nickname or self._current_model
        if target_model is None:
            raise ValueError("No model selected. Use use_model() first.")

        self.pool.activate_adapter(target_model, adapter_id)
        self._current_adapter = adapter_id
        return self

    def use_base_model(self, model_nickname: Optional[str] = None) -> 'SIERAModel':
        """
        Switch to using the base model (no adapter).

        Args:
            model_nickname: Model to deactivate adapter on (uses current if None)

        Returns:
            Self for chaining
        """
        target_model = model_nickname or self._current_model
        if target_model is None:
            raise ValueError("No model selected. Use use_model() first.")

        self.pool.deactivate_adapter(target_model)
        self._current_adapter = None
        return self

    def get_model(self) -> torch.nn.Module:
        """Get the current PyTorch model."""
        if self._current_model is None:
            raise ValueError("No model selected. Use use_model() first.")

        manager = self.pool.get_model(self._current_model)
        return manager.get_model()

    def get_tokenizer(self) -> AutoTokenizer:
        """Get the current tokenizer."""
        if self._current_model is None:
            raise ValueError("No model selected. Use use_model() first.")

        manager = self.pool.get_model(self._current_model)
        return manager.get_tokenizer()

    def get_manager(self) -> BaseModelManager:
        """Get the current model manager (for advanced operations)."""
        if self._current_model is None:
            raise ValueError("No model selected. Use use_model() first.")

        return self.pool.get_model(self._current_model)

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text using the current model and adapter.

        Args:
            prompt: Input prompt
            max_length: Maximum total length (ignored for vLLM)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (ignored for vLLM)
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        manager = self.get_manager()

        # Use the manager's generate method which handles both backends
        result = manager.generate(
            prompts=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_adapter=self._current_adapter,
            **kwargs
        )

        return result

    def chat(
        self,
        messages: List[Dict[str, str]],
        **generation_kwargs
    ) -> str:
        """
        Generate a chat response using the model's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **generation_kwargs: Arguments passed to generate()

        Returns:
            Generated response
        """
        tokenizer = self.get_tokenizer()

        # Apply chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            prompt = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    prompt += f"User: {content}\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n"
                elif role == 'system':
                    prompt += f"System: {content}\n"
            prompt += "Assistant: "

        return self.generate(prompt, **generation_kwargs)

    def get_token_priors(
        self,
        prompt: str,
        target_tokens: Optional[List[str]] = None,
        top_k: int = 50,
        temperature: float = 1.0,
        use_adapter: Optional[bool] = None
    ) -> PriorExtractionResult:
        """
        Extract the model's prior probability distribution over tokens at the end of a prompt.

        This is useful for probing what the model "believes" before generation, e.g.:
        - "I flip a coin, and it lands on" -> probabilities for "heads" vs "tails"
        - "The answer is" -> probabilities for "yes" vs "no"

        Args:
            prompt: Input prompt (without completion)
            target_tokens: Optional list of specific tokens to look for (e.g., ["heads", "tails"])
                          If None, returns all top_k tokens
            top_k: Number of top probability tokens to retrieve (1-100 recommended)
                   Note: vLLM only returns top-k, not full vocabulary
            temperature: Sampling temperature (use 1.0 for unbiased probabilities)
            use_adapter: Whether to use the current adapter.
                        If None, uses current adapter if one is active.
                        If True, uses current adapter (raises error if none active).
                        If False, uses base model.

        Returns:
            PriorExtractionResult with token probabilities

        Example:
            >>> model = SIERAModel()
            >>> model.use_model("qwen2.5-3b")
            >>> result = model.get_token_priors(
            ...     prompt="I flip a coin, and it lands on",
            ...     target_tokens=["heads", "tails"]
            ... )
            >>> print(f"P(heads) = {result.target_priors['heads'].probability:.4f}")
            >>> print(f"P(tails) = {result.target_priors['tails'].probability:.4f}")
        """
        manager = self.get_manager()

        # Determine which adapter to use
        if use_adapter is None:
            adapter_id = self._current_adapter
        elif use_adapter:
            if self._current_adapter is None:
                raise ValueError("No adapter active. Use use_adapter() first or set use_adapter=False.")
            adapter_id = self._current_adapter
        else:
            adapter_id = None

        return manager.get_token_priors(
            prompt=prompt,
            target_tokens=target_tokens,
            top_k=top_k,
            temperature=temperature,
            use_adapter=adapter_id
        )

    def get_token_priors_batch(
        self,
        prompts: List[str],
        target_tokens: Optional[List[str]] = None,
        top_k: int = 50,
        temperature: float = 1.0,
        use_adapter: Optional[bool] = None,
    ) -> "List[PriorExtractionResult]":
        """
        Batched analogue of ``get_token_priors``.

        Submits all prompts in a single backend call so vLLM's continuous
        batching can process them efficiently. Returns a list of
        PriorExtractionResult in the same order as ``prompts``.

        See ``get_token_priors`` for the per-prompt semantics. The
        ``use_adapter`` argument follows the same convention (None = current
        adapter, True = current adapter required, False = base model).
        """
        manager = self.get_manager()

        # Determine which adapter to use (mirrors get_token_priors).
        if use_adapter is None:
            adapter_id = self._current_adapter
        elif use_adapter:
            if self._current_adapter is None:
                raise ValueError("No adapter active. Use use_adapter() first or set use_adapter=False.")
            adapter_id = self._current_adapter
        else:
            adapter_id = None

        return manager.get_token_priors_batch(
            prompts=prompts,
            target_tokens=target_tokens,
            top_k=top_k,
            temperature=temperature,
            use_adapter=adapter_id,
        )

    @contextmanager
    def temporary_adapter(self, adapter_id: str, model_nickname: Optional[str] = None):
        """
        Context manager to temporarily use an adapter.

        Args:
            adapter_id: Adapter to temporarily activate
            model_nickname: Model to use (uses current if None)
        """
        target_model = model_nickname or self._current_model
        if target_model is None:
            raise ValueError("No model selected. Use use_model() first.")

        original_adapter = self._current_adapter

        try:
            self.use_adapter(adapter_id, target_model)
            yield self
        finally:
            if original_adapter is not None:
                self.use_adapter(original_adapter, target_model)
            else:
                self.use_base_model(target_model)

    def save_adapter(
        self,
        adapter_id: str,
        save_path: str,
        model_nickname: Optional[str] = None
    ) -> None:
        """
        Save an adapter to disk.

        Args:
            adapter_id: Adapter to save
            save_path: Path to save adapter
            model_nickname: Model containing adapter (uses current if None)
        """
        target_model = model_nickname or self._current_model
        if target_model is None:
            raise ValueError("No model selected. Use use_model() first.")

        self.pool.save_adapter(target_model, adapter_id, save_path)

    def create_adapter(
        self,
        adapter_id: str,
        r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        model_nickname: Optional[str] = None
    ) -> 'SIERAModel':
        """
        Create a new LoRA adapter.

        Args:
            adapter_id: Unique identifier for the new adapter
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Modules to apply LoRA to (uses default if None)
            model_nickname: Model to create adapter for (uses current if None)

        Returns:
            Self for chaining
        """
        target_model = model_nickname or self._current_model
        if target_model is None:
            raise ValueError("No model selected. Use use_model() first.")

        # Get model family for default target modules
        model_config = self.pool.config_loader.get_model_config(target_model)

        if target_modules is None:
            lora_config = self.pool.config_loader.get_lora_config(model_config.family)
            target_modules = lora_config.target_modules

        lora_config = LoRAConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )

        self.pool.load_adapter(
            model_nickname=target_model,
            adapter_id=adapter_id,
            lora_config=lora_config
        )

        return self

    def list_adapters(self, model_nickname: Optional[str] = None) -> List[str]:
        """
        List adapters for a model.

        Args:
            model_nickname: Model to list adapters for (uses current if None)

        Returns:
            List of adapter IDs
        """
        if model_nickname is None:
            # List all adapters across all models
            all_adapters = self.pool.list_all_adapters()
            return [f"{model}:{adapter}" for model, adapters in all_adapters.items() for adapter in adapters]
        else:
            manager = self.pool.get_model(model_nickname)
            return manager.loaded_adapters

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        return {
            "models": self.pool.get_memory_usage(),
            "system": self.pool.get_system_memory()
        }

    def cleanup(self, max_age_seconds: int = 3600) -> None:
        """Clean up old adapters."""
        self.pool.cleanup(max_age_seconds)

    def unload_model(self, model_nickname: str) -> None:
        """Unload a specific model."""
        self.pool.unload_model(model_nickname)
        if self._current_model == model_nickname:
            self._current_model = None
            self._current_adapter = None

    def unload_adapter(
        self,
        adapter_id: str,
        model_nickname: Optional[str] = None
    ) -> None:
        """Unload a specific adapter."""
        target_model = model_nickname or self._current_model
        if target_model is None:
            raise ValueError("No model selected. Use use_model() first.")

        self.pool.unload_adapter(target_model, adapter_id)
        if self._current_adapter == adapter_id:
            self._current_adapter = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.pool.__exit__(exc_type, exc_val, exc_tb)