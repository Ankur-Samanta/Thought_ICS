"""
SIERA Model Management System

Provides infrastructure for managing multiple language models and LoRA adapters
with automatic memory management, adapter swapping, and unified interfaces.
"""

# Force vLLM to use v0 engine (v1 has multiprocessing issues with tensor parallelism)
import os
os.environ.setdefault('VLLM_USE_V1', '0')

from .interface import SIERAModel
from .pool import ModelPool
from .base_manager import BaseModelManager
from .config import (
    ModelConfig,
    LoRAConfig,
    AdapterInfo,
    MemoryConfig,
    ModelConfigLoader,
    TokenPrior,
    PriorExtractionResult
)

__all__ = [
    "SIERAModel",
    "ModelPool",
    "BaseModelManager",
    "ModelConfig",
    "LoRAConfig",
    "AdapterInfo",
    "MemoryConfig",
    "ModelConfigLoader",
    "TokenPrior",
    "PriorExtractionResult"
]