"""
Base model manager for individual model instances.
"""
import gc
import os
import shutil
import time
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Lazy imports to avoid initializing CUDA before vLLM multiprocessing
# torch, transformers, and peft are imported only when needed
import torch

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    LoRARequest = None

from .config import ModelConfig, LoRAConfig, AdapterInfo, InferenceConfig, TokenPrior, PriorExtractionResult

logger = logging.getLogger(__name__)


class BaseModelManager:
    """Manages a single base model and its LoRA adapters."""

    def __init__(
        self,
        model_config: ModelConfig,
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        inference_config: Optional[InferenceConfig] = None
    ):
        self.model_config = model_config
        # Defer CUDA check to avoid initializing CUDA before vLLM multiprocessing
        self.device = device or "cuda"
        self.quantization = quantization
        self.inference_config = inference_config or InferenceConfig()

        # Backend selection
        self.use_vllm = self.inference_config.backend == "vllm" and VLLM_AVAILABLE
        if self.inference_config.backend == "vllm" and not VLLM_AVAILABLE:
            logger.warning("vLLM requested but not available. Falling back to transformers.")
            self.use_vllm = False

        # Model and tokenizer
        self._base_model = None  # For transformers backend
        self._vllm_model = None  # For vLLM backend
        self._tokenizer = None
        self._current_model = None  # Either base model or PEFT model

        # Adapter management
        self._adapters: Dict[str, AdapterInfo] = {}
        self._active_adapter: Optional[str] = None
        self._peft_model: Optional[PeftModel] = None

        # vLLM adapter tracking
        self._vllm_lora_paths: Dict[str, str] = {}  # adapter_id -> path
        self._adapter_int_ids: Dict[str, int] = {}  # adapter_id -> stable int ID for vLLM
        self._next_adapter_id: int = 1  # Counter for stable adapter IDs

        # Default adapter for generate() when use_adapter=None
        self._default_adapter: Optional[str] = None

        # State tracking
        self._loaded_at = time.time()
        self._last_used = time.time()
        self._memory_usage = 0

    @property
    def is_loaded(self) -> bool:
        """Check if the base model is loaded."""
        if self.use_vllm:
            return self._vllm_model is not None
        return self._base_model is not None

    @property
    def active_adapter(self) -> Optional[str]:
        """Get the currently active adapter ID."""
        return self._active_adapter

    @property
    def loaded_adapters(self) -> List[str]:
        """Get list of loaded adapter IDs."""
        return list(self._adapters.keys())

    @property
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        return self._memory_usage

    def load_base_model(self) -> None:
        """Load the base model and tokenizer."""
        if self.is_loaded:
            logger.info(f"Model {self.model_config.nickname} already loaded")
            return

        logger.info(f"Loading base model: {self.model_config.hf_name} (backend: {'vllm' if self.use_vllm else 'transformers'})")

        if self.use_vllm:
            self._load_vllm_model()
        else:
            self._load_transformers_model()

        # Load tokenizer (shared between backends) - lazy import
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.hf_name,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        # Ensure pad token exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._update_memory_usage()
        logger.info(f"Model {self.model_config.nickname} loaded successfully")

    def _load_transformers_model(self) -> None:
        """Load model using HuggingFace Transformers."""
        # Lazy import to avoid CUDA initialization before vLLM multiprocessing
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        # Setup quantization if requested
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model with appropriate device placement
        # Device mapping strategy depends on inference config:
        # - force_single_gpu=True: Use device_map="cuda:0" (for TTT training)
        # - allow_multi_gpu=True: Use device_map=None (for multi-GPU training with DDP)
        # - explicit device set: Use that device
        # Note: device_map="auto" causes model sharding which breaks standard training

        if self.inference_config.force_single_gpu:
            # Force single GPU placement (for TTT training)
            device_map_value = "cuda:0" if self.device == "cuda" else None
        elif self.inference_config.device:
            # Use explicit device from config
            device_map_value = self.inference_config.device
        elif self.inference_config.allow_multi_gpu:
            # Allow multi-GPU training (PyTorch will handle DataParallel/DDP)
            device_map_value = None
        else:
            # Default behavior
            device_map_value = "cuda:0" if self.device == "cuda" else None

        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.hf_name,
            torch_dtype=self.model_config.get_torch_dtype(),
            device_map=device_map_value,
            trust_remote_code=self.model_config.trust_remote_code,
            quantization_config=quantization_config,
        )
        self._current_model = self._base_model

    def _load_vllm_model(self) -> None:
        """Load model using vLLM."""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. Install with: pip install vllm")

        # Build vLLM initialization arguments
        vllm_kwargs = {
            "model": self.model_config.hf_name,
            "trust_remote_code": self.inference_config.trust_remote_code,
            "dtype": self.inference_config.dtype,
            "gpu_memory_utilization": self.inference_config.gpu_memory_utilization,
            "tensor_parallel_size": self.inference_config.tensor_parallel_size,
            "pipeline_parallel_size": self.inference_config.pipeline_parallel_size,
            "max_num_seqs": self.inference_config.max_num_seqs,
            "enforce_eager": self.inference_config.enforce_eager,
        }

        # Add optional parameters
        # Use explicit max_model_len from inference_config, or fall back to model's context_length
        if self.inference_config.max_model_len is not None:
            vllm_kwargs["max_model_len"] = self.inference_config.max_model_len
        elif self.model_config.context_length is not None:
            vllm_kwargs["max_model_len"] = self.model_config.context_length

        # Add model seed if specified (for deterministic generation)
        if self.inference_config.model_seed is not None:
            vllm_kwargs["seed"] = self.inference_config.model_seed

        # Enable LoRA support if configured
        if self.inference_config.enable_lora:
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_loras"] = self.inference_config.max_loras
            vllm_kwargs["max_lora_rank"] = self.inference_config.max_lora_rank
            if self.inference_config.max_cpu_loras is not None:
                vllm_kwargs["max_cpu_loras"] = self.inference_config.max_cpu_loras

        self._vllm_model = LLM(**vllm_kwargs)

    def unload_base_model(self) -> None:
        """Unload the base model and all adapters."""
        if not self.is_loaded:
            return

        logger.info(f"Unloading model: {self.model_config.nickname}")

        # Clear all adapters first
        self.clear_all_adapters()

        # Clear model references
        self._base_model = None
        self._vllm_model = None
        self._tokenizer = None
        self._current_model = None
        self._memory_usage = 0

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_adapter(
        self,
        adapter_id: str,
        adapter_path: Optional[str] = None,
        lora_config: Optional[LoRAConfig] = None,
        **metadata
    ) -> None:
        """Load a LoRA adapter."""
        if not self.is_loaded:
            raise RuntimeError("Base model must be loaded before loading adapters")

        if adapter_id in self._adapters:
            logger.info(f"Adapter {adapter_id} already loaded")
            return

        logger.info(f"Loading LoRA adapter: {adapter_id}")

        try:
            if self.use_vllm:
                self._load_vllm_adapter(adapter_id, adapter_path, lora_config, metadata)
            else:
                self._load_transformers_adapter(adapter_id, adapter_path, lora_config, metadata)

            logger.info(f"Adapter {adapter_id} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_id}: {e}")
            raise

    def _load_transformers_adapter(
        self,
        adapter_id: str,
        adapter_path: Optional[str],
        lora_config: Optional[LoRAConfig],
        metadata: Dict[str, Any]
    ) -> None:
        """Load adapter using Transformers/PEFT."""
        # Lazy import
        from peft import PeftModel, LoraConfig as PeftLoraConfig, get_peft_model, TaskType

        if adapter_path:
            # Load existing adapter from path
            if self._peft_model is None:
                self._peft_model = PeftModel.from_pretrained(
                    self._base_model,
                    adapter_path,
                    adapter_name=adapter_id
                )
            else:
                self._peft_model.load_adapter(adapter_path, adapter_name=adapter_id)

            # Get config from loaded adapter
            adapter_config = self._peft_model.peft_config[adapter_id]
            # task_type might be enum or string depending on how adapter was saved
            task_type = adapter_config.task_type
            if hasattr(task_type, 'value'):
                task_type = task_type.value
            lora_config = LoRAConfig(
                r=adapter_config.r,
                lora_alpha=adapter_config.lora_alpha,
                lora_dropout=adapter_config.lora_dropout,
                bias=adapter_config.bias,
                task_type=task_type,
                target_modules=adapter_config.target_modules
            )
        else:
            # Create new adapter
            if lora_config is None:
                raise ValueError("lora_config required when not loading from path")

            peft_config = PeftLoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                task_type=TaskType.CAUSAL_LM,
                target_modules=lora_config.target_modules,
            )

            if self._peft_model is None:
                self._peft_model = get_peft_model(self._base_model, peft_config)
                # The first adapter gets a default name, rename it
                self._peft_model.add_adapter(adapter_id, peft_config)
                self._peft_model.delete_adapter("default")
            else:
                self._peft_model.add_adapter(adapter_id, peft_config)

        # Track adapter info
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            config=lora_config,
            loaded_at=time.time(),
            last_used=time.time(),
            memory_usage=self._estimate_adapter_memory(lora_config),
            metadata=metadata
        )
        self._adapters[adapter_id] = adapter_info

        # Update current model reference
        self._current_model = self._peft_model

    def _load_vllm_adapter(
        self,
        adapter_id: str,
        adapter_path: Optional[str],
        lora_config: Optional[LoRAConfig],
        metadata: Dict[str, Any]
    ) -> None:
        """Load adapter for vLLM."""
        if adapter_path is None:
            raise ValueError("vLLM requires adapter_path - cannot create new adapters at runtime")

        # Validate adapter files exist before storing path
        adapter_path_obj = Path(adapter_path)
        if not adapter_path_obj.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        # Check for required adapter weights file (safetensors or bin format)
        safetensors_path = adapter_path_obj / "adapter_model.safetensors"
        bin_path = adapter_path_obj / "adapter_model.bin"
        if not safetensors_path.exists() and not bin_path.exists():
            raise FileNotFoundError(
                f"Missing adapter weights file in {adapter_path}. "
                f"Expected 'adapter_model.safetensors' or 'adapter_model.bin'"
            )

        # Store adapter path for vLLM LoRARequest
        self._vllm_lora_paths[adapter_id] = adapter_path

        # Try to infer config if not provided
        if lora_config is None:
            # Try to load config from adapter path
            config_path = Path(adapter_path) / "adapter_config.json"
            if config_path.exists():
                import json
                with open(config_path) as f:
                    adapter_config = json.load(f)
                lora_config = LoRAConfig(
                    r=adapter_config.get("r", 32),
                    lora_alpha=adapter_config.get("lora_alpha", 64),
                    lora_dropout=adapter_config.get("lora_dropout", 0.0),
                    bias=adapter_config.get("bias", "none"),
                    task_type=adapter_config.get("task_type", "CAUSAL_LM"),
                    target_modules=adapter_config.get("target_modules", [])
                )
            else:
                # Use default config
                lora_config = LoRAConfig(
                    r=32, lora_alpha=64, lora_dropout=0.0,
                    bias="none", task_type="CAUSAL_LM", target_modules=[]
                )

        # Track adapter info
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            config=lora_config,
            loaded_at=time.time(),
            last_used=time.time(),
            memory_usage=self._estimate_adapter_memory(lora_config),
            metadata=metadata
        )
        self._adapters[adapter_id] = adapter_info

    def unload_adapter(self, adapter_id: str) -> None:
        """Unload a specific LoRA adapter."""
        if adapter_id not in self._adapters:
            logger.warning(f"Adapter {adapter_id} not found")
            return

        logger.info(f"Unloading adapter: {adapter_id}")

        # If this is the active adapter, deactivate it
        if self._active_adapter == adapter_id:
            self.deactivate_adapter()

        # Remove from PEFT model
        if self._peft_model is not None:
            self._peft_model.delete_adapter(adapter_id)

            # If no adapters left, switch back to base model
            if len(self._peft_model.peft_config) == 0:
                self._current_model = self._base_model
                self._peft_model = None

        # Remove from tracking
        del self._adapters[adapter_id]

        logger.info(f"Adapter {adapter_id} unloaded")

    def activate_adapter(self, adapter_id: str) -> None:
        """Activate a specific LoRA adapter."""
        if adapter_id not in self._adapters:
            raise ValueError(f"Adapter {adapter_id} not loaded")

        if self._active_adapter == adapter_id:
            logger.info(f"Adapter {adapter_id} already active")
            return

        logger.info(f"Activating adapter: {adapter_id}")

        if self._peft_model is not None:
            self._peft_model.set_adapter(adapter_id)

        self._active_adapter = adapter_id
        self._adapters[adapter_id].last_used = time.time()
        self._last_used = time.time()

    def deactivate_adapter(self) -> None:
        """Deactivate the current adapter (use base model)."""
        if self._active_adapter is None:
            return

        logger.info(f"Deactivating adapter: {self._active_adapter}")

        if self._peft_model is not None:
            try:
                self._peft_model.disable_adapters()
            except ValueError as e:
                # Adapter already disabled or not loaded - this is okay
                logger.debug(f"Could not disable adapter (may already be disabled): {e}")

        self._active_adapter = None
        self._last_used = time.time()

    def set_default_adapter(self, adapter_id: Optional[str]) -> None:
        """Set the default adapter to use when use_adapter=None in generate().

        This allows external code (like ICTS pipeline) to use adapters without
        modification. Pass None to use base model by default.

        When setting a new adapter, the previous default adapter is cleaned up
        (removed from tracking, files deleted if on disk). vLLM handles GPU
        memory eviction internally via max_loras.

        Args:
            adapter_id: Adapter ID to use as default, or None for base model
        """
        # Clean up previous default adapter if it exists and is different
        if self._default_adapter is not None and self._default_adapter != adapter_id:
            old_adapter = self._default_adapter
            logger.info(f"Cleaning up previous default adapter: {old_adapter}")

            # Remove from tracking dicts
            if old_adapter in self._adapters:
                del self._adapters[old_adapter]
            if old_adapter in self._vllm_lora_paths:
                old_path = self._vllm_lora_paths[old_adapter]
                del self._vllm_lora_paths[old_adapter]
                # Delete files from disk
                if os.path.exists(old_path):
                    shutil.rmtree(old_path)
                    logger.info(f"Deleted adapter files: {old_path}")
            if old_adapter in self._adapter_int_ids:
                del self._adapter_int_ids[old_adapter]

        # Validate new adapter exists (if not None)
        if adapter_id is not None and adapter_id not in self._adapters:
            raise ValueError(f"Adapter {adapter_id} not loaded")

        self._default_adapter = adapter_id
        logger.info(f"Default adapter set to: {adapter_id or 'base model'}")

    @property
    def default_adapter(self) -> Optional[str]:
        """Get the current default adapter."""
        return self._default_adapter

    def clear_all_adapters(self) -> None:
        """Remove all loaded adapters."""
        adapter_ids = list(self._adapters.keys())
        for adapter_id in adapter_ids:
            self.unload_adapter(adapter_id)

    def save_adapter(self, adapter_id: str, save_path: str) -> None:
        """Save a specific adapter to disk."""
        if adapter_id not in self._adapters:
            raise ValueError(f"Adapter {adapter_id} not loaded")

        logger.info(f"Saving adapter {adapter_id} to {save_path}")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self._peft_model is not None:
            self._peft_model.save_pretrained(
                save_path,
                selected_adapters=[adapter_id]
            )

        # Update adapter info with save path
        self._adapters[adapter_id].adapter_path = str(save_path)

    def get_adapter_info(self, adapter_id: str) -> Optional[AdapterInfo]:
        """Get information about a specific adapter."""
        return self._adapters.get(adapter_id)

    def list_adapters(self) -> Dict[str, AdapterInfo]:
        """Get information about all loaded adapters."""
        return self._adapters.copy()

    def get_model(self) -> torch.nn.Module:
        """Get the current model (base or with active adapter)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        if self.use_vllm:
            return self._vllm_model
        return self._current_model

    def get_tokenizer(self):
        """Get the tokenizer."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return self._tokenizer

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_adapter: Optional[str] = None,
        n: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text using the appropriate backend.

        Args:
            prompts: Single prompt string or list of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            use_adapter: Adapter ID to use (None uses default_adapter if set, otherwise base model)
            n: Number of completions to generate per prompt (default: 1)
               When n > 1, returns n completions for each prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text(s). When n=1, returns one text per prompt.
            When n>1, returns n texts per prompt (flattened list).
            For single prompt input with n=1, returns a single string.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]

        # Use default adapter if none specified
        if use_adapter is None:
            use_adapter = self._default_adapter

        if self.use_vllm:
            results = self._generate_vllm(prompts, max_tokens, temperature, top_p, top_k, use_adapter, n, **kwargs)
        else:
            results = self._generate_transformers(prompts, max_tokens, temperature, top_p, top_k, use_adapter, n, **kwargs)

        # Return format depends on n and is_single
        if is_single and n == 1:
            return results[0]
        else:
            return results

    def _generate_transformers(
        self,
        prompts: List[str],
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
        use_adapter: Optional[str],
        n: int,
        **kwargs
    ) -> List[str]:
        """Generate using Transformers backend.

        Args:
            n: Number of completions to generate per prompt

        Returns:
            List of generated texts. When n > 1, returns n texts per prompt
            in flattened order: [prompt1_gen1, prompt1_gen2, ..., prompt2_gen1, ...]
        """
        # Temporarily switch adapter if requested
        original_adapter = self._active_adapter
        if use_adapter is not None and use_adapter != original_adapter:
            if use_adapter in self._adapters:
                self.activate_adapter(use_adapter)
            else:
                logger.warning(f"Adapter {use_adapter} not found, using current adapter")

        try:
            model = self._current_model or self._base_model
            results = []

            for prompt in prompts:
                # Tokenize
                inputs = self._tokenizer(prompt, return_tensors="pt")
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Generate n completions for this prompt
                with torch.no_grad():
                    gen_kwargs = {
                        "temperature": temperature,
                        "top_p": top_p,
                        "do_sample": temperature > 0,
                        "pad_token_id": self._tokenizer.eos_token_id,
                        "num_return_sequences": n,  # Generate n sequences
                    }
                    if max_tokens is not None:
                        gen_kwargs["max_new_tokens"] = max_tokens
                    if top_k > 0:
                        gen_kwargs["top_k"] = top_k
                    gen_kwargs.update(kwargs)

                    outputs = model.generate(**inputs, **gen_kwargs)

                # Decode all n generated sequences
                input_length = inputs['input_ids'].shape[1]
                for output_seq in outputs:
                    generated_tokens = output_seq[input_length:]
                    generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    results.append(generated_text)

            return results

        finally:
            # Restore original adapter
            if use_adapter is not None and use_adapter != original_adapter:
                if original_adapter is not None:
                    self.activate_adapter(original_adapter)
                else:
                    self.deactivate_adapter()

    def _generate_vllm(
        self,
        prompts: List[str],
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
        use_adapter: Optional[str],
        n: int,
        **kwargs
    ) -> List[str]:
        """Generate using vLLM backend.

        Args:
            n: Number of completions to generate per prompt

        Returns:
            List of generated texts. When n > 1, returns n texts per prompt
            in flattened order: [prompt1_gen1, prompt1_gen2, ..., prompt2_gen1, ...]
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available")

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=max_tokens or 512,
            n=n,  # Number of completions per prompt
        )

        # Add any additional sampling params from kwargs
        for key in ['frequency_penalty', 'presence_penalty', 'repetition_penalty', 'length_penalty', 'stop', 'stop_token_ids', 'min_tokens', 'best_of']:
            if key in kwargs:
                setattr(sampling_params, key, kwargs[key])

        # Create LoRA request if adapter specified
        lora_request = None
        if use_adapter is not None:
            if use_adapter not in self._vllm_lora_paths:
                raise ValueError(f"Adapter {use_adapter} not loaded")
            # Use stable sequential ID instead of hash (hash is non-deterministic across processes)
            if use_adapter not in self._adapter_int_ids:
                self._adapter_int_ids[use_adapter] = self._next_adapter_id
                self._next_adapter_id += 1
            lora_request = LoRARequest(
                lora_name=use_adapter,
                lora_int_id=self._adapter_int_ids[use_adapter],
                lora_local_path=self._vllm_lora_paths[use_adapter]
            )
            # Update last used
            if use_adapter in self._adapters:
                self._adapters[use_adapter].last_used = time.time()

        # Generate
        outputs = self._vllm_model.generate(
            prompts,
            sampling_params,
            lora_request=lora_request
        )

        # Extract generated texts
        # When n > 1, each RequestOutput contains multiple CompletionOutput objects
        results = []
        for request_output in outputs:
            # request_output.outputs is a list of CompletionOutput (length = n)
            for completion_output in request_output.outputs:
                results.append(completion_output.text)

        return results

    def get_token_priors(
        self,
        prompt: str,
        target_tokens: Optional[List[str]] = None,
        top_k: int = 50,
        temperature: float = 1.0,
        use_adapter: Optional[str] = None,
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
            use_adapter: Adapter ID to use (None for base model)

        Returns:
            PriorExtractionResult with token probabilities

        Note:
            - Automatically checks tokenization variants (with/without leading space)
            - Only supports single-token completions (multi-token support can be added later)
            - If a target token isn't in top_k, it won't appear in results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        if self.use_vllm:
            return self._get_token_priors_vllm(
                prompt, target_tokens, top_k, temperature, use_adapter
            )
        else:
            return self._get_token_priors_transformers(
                prompt, target_tokens, top_k, temperature, use_adapter
            )

    def get_token_priors_batch(
        self,
        prompts: List[str],
        target_tokens: Optional[List[str]] = None,
        top_k: int = 50,
        temperature: float = 1.0,
        use_adapter: Optional[str] = None,
    ) -> List[PriorExtractionResult]:
        """
        Extract token priors for a list of prompts in a single batched call.

        This is the batched analogue of ``get_token_priors``. The returned list
        is in the same order as ``prompts``. Each entry has the same shape as
        ``get_token_priors`` would return for that prompt in isolation.

        For the vLLM backend, all prompts are submitted in one
        ``LLM.generate`` call so vLLM's continuous batching scheduler can
        process them with maximum efficiency.

        For the Transformers backend, prompts are processed sequentially via
        the existing single-prompt path (no padded batching is implemented;
        the win for batching is concentrated in the vLLM path).

        Args:
            prompts: List of input prompts (each without completion).
            target_tokens: Optional list of target tokens to look for in each
                prompt's distribution. The same target list is used for every
                prompt.
            top_k: Number of top probability tokens to retrieve per prompt.
            temperature: Sampling temperature.
            use_adapter: Adapter ID to use, or None for the base model.

        Returns:
            List of PriorExtractionResult, one per input prompt, in input
            order.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        if self.use_vllm:
            return self._get_token_priors_vllm_batch(
                prompts, target_tokens, top_k, temperature, use_adapter
            )

        # Transformers fallback: sequential single-prompt calls.
        return [
            self._get_token_priors_transformers(
                p, target_tokens, top_k, temperature, use_adapter
            )
            for p in prompts
        ]

    @staticmethod
    def _parse_vllm_prior_output(
        prompt: str,
        vllm_output,
        target_tokens: Optional[List[str]],
        top_k: int,
    ) -> PriorExtractionResult:
        """Parse a single vLLM RequestOutput into a PriorExtractionResult.

        Factored out so that both the single-prompt and batched code paths
        share identical parsing logic. Pure function: depends only on its
        arguments, no model or class state.
        """
        import math

        # Extract logprobs from first token position
        first_token_logprobs = vllm_output.outputs[0].logprobs[0]  # Dict[int, Logprob]

        # Build top_tokens list (all top-k tokens sorted by probability)
        top_tokens_list = []
        for token_id, logprob_obj in first_token_logprobs.items():
            probability = math.exp(logprob_obj.logprob)
            token_prior = TokenPrior(
                token_text=logprob_obj.decoded_token,
                token_id=token_id,
                logprob=logprob_obj.logprob,
                probability=probability,
                rank=logprob_obj.rank
            )
            top_tokens_list.append(token_prior)

        # Sort by probability descending
        top_tokens_list.sort(key=lambda x: x.probability, reverse=True)

        # If no target tokens specified, return all top-k
        if target_tokens is None:
            return PriorExtractionResult(
                prompt=prompt,
                target_priors={},
                top_tokens=top_tokens_list,
                found_all_targets=True
            )

        # Find target tokens in the logprobs
        # Need to check tokenization variants (with/without leading space, capitalization)
        target_priors = {}
        found_count = 0

        for target in target_tokens:
            # Check all variants
            variants = [
                target,
                " " + target,  # With leading space
                target.capitalize(),
                " " + target.capitalize(),
                target.lower(),
                " " + target.lower(),
                target.upper(),
                " " + target.upper(),
            ]

            # Remove duplicates while preserving order
            seen = set()
            unique_variants = []
            for v in variants:
                if v not in seen:
                    seen.add(v)
                    unique_variants.append(v)

            # Search for any variant in the top_tokens
            best_match = None
            best_prob = 0.0

            for token_prior in top_tokens_list:
                token_text = token_prior.token_text
                # Check exact match with any variant
                if token_text in unique_variants:
                    if token_prior.probability > best_prob:
                        best_match = token_prior
                        best_prob = token_prior.probability

            if best_match is not None:
                target_priors[target] = best_match
                found_count += 1
            else:
                logger.warning(
                    f"Target token '{target}' not found in top-{top_k} tokens. "
                    f"Consider increasing top_k or checking tokenization."
                )

        found_all = found_count == len(target_tokens)

        return PriorExtractionResult(
            prompt=prompt,
            target_priors=target_priors,
            top_tokens=top_tokens_list,
            found_all_targets=found_all
        )

    def _build_prior_lora_request(self, use_adapter: Optional[str]):
        """Build a LoRARequest for prior extraction (or return None).

        Shared by single-prompt and batched code paths.
        """
        if use_adapter is None:
            return None
        if use_adapter not in self._vllm_lora_paths:
            raise ValueError(f"Adapter {use_adapter} not loaded")
        # Use stable sequential ID instead of hash (hash is non-deterministic across processes)
        if use_adapter not in self._adapter_int_ids:
            self._adapter_int_ids[use_adapter] = self._next_adapter_id
            self._next_adapter_id += 1
        lora_request = LoRARequest(
            lora_name=use_adapter,
            lora_int_id=self._adapter_int_ids[use_adapter],
            lora_local_path=self._vllm_lora_paths[use_adapter]
        )
        if use_adapter in self._adapters:
            self._adapters[use_adapter].last_used = time.time()
        return lora_request

    def _get_token_priors_vllm(
        self,
        prompt: str,
        target_tokens: Optional[List[str]],
        top_k: int,
        temperature: float,
        use_adapter: Optional[str],
    ) -> PriorExtractionResult:
        """Get token priors using vLLM backend."""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available")

        # Create sampling params for prior extraction
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,  # Don't restrict vocabulary
            top_k=-1,   # Don't restrict vocabulary
            max_tokens=1,  # Only need first token
            logprobs=top_k,  # Get top-k token probabilities
        )

        lora_request = self._build_prior_lora_request(use_adapter)

        # Generate with logprobs
        outputs = self._vllm_model.generate(
            [prompt],
            sampling_params,
            lora_request=lora_request
        )

        return self._parse_vllm_prior_output(prompt, outputs[0], target_tokens, top_k)

    def _get_token_priors_vllm_batch(
        self,
        prompts: List[str],
        target_tokens: Optional[List[str]],
        top_k: int,
        temperature: float,
        use_adapter: Optional[str],
    ) -> List[PriorExtractionResult]:
        """Get token priors for a batch of prompts using vLLM backend.

        vLLM's continuous batching handles scheduling internally. The list of
        returned PriorExtractionResults is in the same order as ``prompts``.
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available")

        if not prompts:
            return []

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            logprobs=top_k,
        )

        lora_request = self._build_prior_lora_request(use_adapter)

        outputs = self._vllm_model.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
        )

        # vLLM preserves input order in its returned outputs.
        return [
            self._parse_vllm_prior_output(prompt, out, target_tokens, top_k)
            for prompt, out in zip(prompts, outputs)
        ]

    def _get_token_priors_transformers(
        self,
        prompt: str,
        target_tokens: Optional[List[str]],
        top_k: int,
        temperature: float,
        use_adapter: Optional[str],
    ) -> PriorExtractionResult:
        """Get token priors using Transformers backend."""
        import math
        import torch.nn.functional as F

        # Temporarily switch adapter if requested
        original_adapter = self._active_adapter
        if use_adapter is not None and use_adapter != original_adapter:
            if use_adapter in self._adapters:
                self.activate_adapter(use_adapter)
            else:
                logger.warning(f"Adapter {use_adapter} not found, using current adapter")

        try:
            model = self._current_model or self._base_model

            # Tokenize prompt
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Get logits for next token
            with torch.no_grad():
                outputs = model(**inputs)
                next_token_logits = outputs.logits[0, -1, :]  # Last position logits

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)

                # Get top-k tokens
                top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

            # Build top_tokens list
            top_tokens_list = []
            for rank, (prob, token_id) in enumerate(zip(top_probs.cpu().tolist(), top_indices.cpu().tolist())):
                token_text = self._tokenizer.decode([token_id])
                logprob = math.log(prob) if prob > 0 else float('-inf')

                token_prior = TokenPrior(
                    token_text=token_text,
                    token_id=token_id,
                    logprob=logprob,
                    probability=prob,
                    rank=rank + 1
                )
                top_tokens_list.append(token_prior)

            # If no target tokens specified, return all top-k
            if target_tokens is None:
                return PriorExtractionResult(
                    prompt=prompt,
                    target_priors={},
                    top_tokens=top_tokens_list,
                    found_all_targets=True
                )

            # Find target tokens (same logic as vLLM)
            target_priors = {}
            found_count = 0

            for target in target_tokens:
                variants = [
                    target,
                    " " + target,
                    target.capitalize(),
                    " " + target.capitalize(),
                    target.lower(),
                    " " + target.lower(),
                    target.upper(),
                    " " + target.upper(),
                ]

                seen = set()
                unique_variants = []
                for v in variants:
                    if v not in seen:
                        seen.add(v)
                        unique_variants.append(v)

                best_match = None
                best_prob = 0.0

                for token_prior in top_tokens_list:
                    token_text = token_prior.token_text
                    if token_text in unique_variants:
                        if token_prior.probability > best_prob:
                            best_match = token_prior
                            best_prob = token_prior.probability

                if best_match is not None:
                    target_priors[target] = best_match
                    found_count += 1
                else:
                    logger.warning(
                        f"Target token '{target}' not found in top-{top_k} tokens."
                    )

            found_all = found_count == len(target_tokens)

            return PriorExtractionResult(
                prompt=prompt,
                target_priors=target_priors,
                top_tokens=top_tokens_list,
                found_all_targets=found_all
            )

        finally:
            # Restore original adapter
            if use_adapter is not None and use_adapter != original_adapter:
                if original_adapter is not None:
                    self.activate_adapter(original_adapter)
                else:
                    self.deactivate_adapter()

    def _estimate_adapter_memory(self, lora_config: LoRAConfig) -> int:
        """Estimate memory usage of a LoRA adapter."""
        # Rough estimation based on LoRA rank and target modules
        # This is a simplified calculation
        num_modules = len(lora_config.target_modules)
        params_per_module = lora_config.r * 4096  # Rough estimate
        total_params = num_modules * params_per_module * 2  # A and B matrices
        bytes_per_param = 2 if self.model_config.torch_dtype == "bfloat16" else 4
        return total_params * bytes_per_param

    def _update_memory_usage(self) -> None:
        """Update estimated memory usage."""
        if not self.is_loaded:
            self._memory_usage = 0
            return

        # Base model memory (rough estimate)
        if self.use_vllm:
            # vLLM reports memory usage differently, use a rough estimate
            # Based on model size (3B params ~= 6GB in bfloat16)
            base_memory = 6 * 1024 * 1024 * 1024  # 6GB estimate
        else:
            base_memory = self._base_model.num_parameters() * 2  # bfloat16

        # Add adapter memory
        adapter_memory = sum(
            adapter.memory_usage for adapter in self._adapters.values()
        )

        self._memory_usage = base_memory + adapter_memory

    def cleanup_old_adapters(self, max_age_seconds: int = 3600) -> None:
        """Remove adapters that haven't been used recently."""
        current_time = time.time()
        to_remove = []

        for adapter_id, adapter_info in self._adapters.items():
            if current_time - adapter_info.last_used > max_age_seconds:
                if adapter_id != self._active_adapter:
                    to_remove.append(adapter_id)

        for adapter_id in to_remove:
            logger.info(f"Cleaning up old adapter: {adapter_id}")
            self.unload_adapter(adapter_id)

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.unload_base_model()
        except:
            pass  # Ignore errors during cleanup