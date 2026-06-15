"""
Model configuration and data structures for SIERA.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
import torch


@dataclass
class ModelConfig:
    """Configuration for a base model."""
    hf_name: str
    nickname: str
    family: str
    size: str
    context_length: int
    trust_remote_code: bool
    torch_dtype: str
    chat_template: bool
    stop_tokens: List[str]

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto"
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    task_type: str
    target_modules: List[str]


@dataclass
class AdapterInfo:
    """Information about a loaded LoRA adapter."""
    adapter_id: str
    adapter_path: Optional[str]
    config: LoRAConfig
    loaded_at: float  # timestamp
    last_used: float  # timestamp
    memory_usage: int  # estimated bytes
    metadata: Dict[str, Any]  # arbitrary metadata

    def __post_init__(self):
        if self.last_used == 0:
            self.last_used = self.loaded_at


@dataclass
class TokenPrior:
    """Information about a token's prior probability."""
    token_text: str
    token_id: int
    logprob: float
    probability: float
    rank: Optional[int] = None


@dataclass
class PriorExtractionResult:
    """Result of extracting token priors from a prompt."""
    prompt: str
    target_priors: Dict[str, TokenPrior]  # target token string -> TokenPrior
    top_tokens: List[TokenPrior]  # All top-k tokens sorted by probability
    found_all_targets: bool  # Whether all target tokens were found in top-k


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    max_models_loaded: int
    max_adapters_per_model: int
    cleanup_threshold: float
    adapter_cache_size: int


@dataclass
class InferenceConfig:
    """Inference backend configuration."""
    backend: str = "vllm"  # "transformers" or "vllm"
    max_num_seqs: int = 256  # vLLM: max number of sequences
    max_model_len: Optional[int] = None  # vLLM: max model length
    gpu_memory_utilization: float = 0.9  # vLLM: GPU memory utilization
    tensor_parallel_size: int = 1  # vLLM: tensor parallelism
    pipeline_parallel_size: int = 1  # vLLM: pipeline parallelism
    trust_remote_code: bool = True  # vLLM: trust remote code
    dtype: str = "auto"  # vLLM: data type
    enforce_eager: bool = False  # vLLM: disable CUDA graphs
    enable_lora: bool = True  # vLLM: enable LoRA support
    max_loras: int = 4  # vLLM: max number of LoRAs
    max_lora_rank: int = 64  # vLLM: max LoRA rank
    max_cpu_loras: Optional[int] = None  # vLLM: max LoRAs in CPU memory
    model_seed: Optional[int] = None  # vLLM: seed for model generation (None=non-deterministic)

    # Transformers backend settings
    device: Optional[str] = None  # Explicit device (e.g., "cuda:0", "cuda")
    force_single_gpu: bool = False  # Force single GPU with device_map="cuda:0"
    allow_multi_gpu: bool = True  # Allow multi-GPU training (DataParallel/DDP)


class ModelConfigLoader:
    """Loads and validates model configurations."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
        self.config_path = Path(config_path)
        self._config_data = None

    def load(self) -> Dict[str, Any]:
        """Load the configuration file."""
        if self._config_data is None:
            with open(self.config_path, 'r') as f:
                self._config_data = yaml.safe_load(f)
        return self._config_data

    def get_model_config(self, nickname: str) -> ModelConfig:
        """Get configuration for a specific model."""
        config_data = self.load()

        if nickname not in config_data["models"]:
            available = list(config_data["models"].keys())
            raise ValueError(f"Model '{nickname}' not found. Available: {available}")

        model_data = config_data["models"][nickname]
        return ModelConfig(**model_data)

    def get_lora_config(self, model_family: str, **overrides) -> LoRAConfig:
        """Get LoRA configuration for a model family."""
        config_data = self.load()

        # Start with default config
        lora_config = config_data["default_lora_config"].copy()

        # Apply family-specific config
        if model_family in config_data["lora_configs"]:
            lora_config.update(config_data["lora_configs"][model_family])

        # Apply any overrides
        lora_config.update(overrides)

        return LoRAConfig(**lora_config)

    def get_memory_config(self) -> MemoryConfig:
        """Get memory management configuration."""
        config_data = self.load()
        return MemoryConfig(**config_data["memory_config"])

    def get_inference_config(self) -> InferenceConfig:
        """Get inference backend configuration."""
        config_data = self.load()
        inference_data = config_data.get("inference_config", {})
        return InferenceConfig(**inference_data)

    def list_available_models(self) -> List[str]:
        """List all available model nicknames."""
        config_data = self.load()
        return list(config_data["models"].keys())

    def get_models_by_family(self, family: str) -> List[str]:
        """Get all models of a specific family."""
        config_data = self.load()
        return [
            nickname for nickname, config in config_data["models"].items()
            if config["family"] == family
        ]

    def get_models_by_size(self, max_size: str) -> List[str]:
        """Get models up to a certain size (rough filtering)."""
        config_data = self.load()

        # Simple size comparison (assumes format like "3B", "8B")
        def size_to_float(size_str: str) -> float:
            if size_str.endswith("B"):
                return float(size_str[:-1])
            return float("inf")

        max_size_val = size_to_float(max_size)

        return [
            nickname for nickname, config in config_data["models"].items()
            if size_to_float(config["size"]) <= max_size_val
        ]