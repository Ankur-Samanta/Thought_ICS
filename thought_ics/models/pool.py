"""
Model pool manager for handling multiple models and their adapters.
"""
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from contextlib import contextmanager

import torch
import psutil

from .config import ModelConfigLoader, ModelConfig, LoRAConfig, MemoryConfig, AdapterInfo, InferenceConfig
from .base_manager import BaseModelManager

logger = logging.getLogger(__name__)


class ModelPool:
    """Manages multiple base models and their LoRA adapters."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        default_device: Optional[str] = None,
        default_quantization: Optional[str] = None,
        inference_config: Optional[InferenceConfig] = None
    ):
        self.config_loader = ModelConfigLoader(config_path)
        self.memory_config = self.config_loader.get_memory_config()
        self.inference_config = inference_config or self.config_loader.get_inference_config()
        # Defer CUDA check to avoid initializing CUDA before vLLM multiprocessing
        self.default_device = default_device or "cuda"
        self.default_quantization = default_quantization

        # Model managers
        self._model_managers: Dict[str, BaseModelManager] = {}
        self._load_order: List[str] = []  # Track loading order for LRU

        # Global adapter tracking (for easy lookup across models)
        self._global_adapter_registry: Dict[str, Tuple[str, str]] = {}  # adapter_id -> (model_nickname, adapter_id)

    @property
    def loaded_models(self) -> List[str]:
        """Get list of loaded model nicknames."""
        return [nickname for nickname, manager in self._model_managers.items() if manager.is_loaded]

    @property
    def available_models(self) -> List[str]:
        """Get list of all available model nicknames."""
        return self.config_loader.list_available_models()

    def load_model(
        self,
        model_nickname: str,
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        force_reload: bool = False
    ) -> BaseModelManager:
        """Load a model by nickname."""
        # Check if model already exists
        if model_nickname in self._model_managers and not force_reload:
            manager = self._model_managers[model_nickname]
            if manager.is_loaded:
                # Update LRU order
                self._update_load_order(model_nickname)
                return manager

        # Get model configuration
        model_config = self.config_loader.get_model_config(model_nickname)

        # Check memory constraints
        self._enforce_memory_limits()

        # Create or get manager
        if model_nickname not in self._model_managers or force_reload:
            device = device or self.default_device
            quantization = quantization or self.default_quantization

            manager = BaseModelManager(
                model_config=model_config,
                device=device,
                quantization=quantization,
                inference_config=self.inference_config
            )
            self._model_managers[model_nickname] = manager
        else:
            manager = self._model_managers[model_nickname]

        # Load the model
        try:
            manager.load_base_model()
            self._update_load_order(model_nickname)
            logger.info(f"Model {model_nickname} loaded successfully")
            return manager
        except Exception as e:
            logger.error(f"Failed to load model {model_nickname}: {e}")
            if model_nickname in self._model_managers:
                del self._model_managers[model_nickname]
            raise

    def unload_model(self, model_nickname: str) -> None:
        """Unload a specific model."""
        if model_nickname not in self._model_managers:
            logger.warning(f"Model {model_nickname} not found")
            return

        manager = self._model_managers[model_nickname]

        # Remove adapters from global registry
        for adapter_id in manager.loaded_adapters:
            global_id = f"{model_nickname}:{adapter_id}"
            if global_id in self._global_adapter_registry:
                del self._global_adapter_registry[global_id]

        # Unload the model
        manager.unload_base_model()

        # Remove from load order
        if model_nickname in self._load_order:
            self._load_order.remove(model_nickname)

        logger.info(f"Model {model_nickname} unloaded")

    def get_model(self, model_nickname: str) -> BaseModelManager:
        """Get a model manager by nickname."""
        if model_nickname not in self._model_managers:
            raise ValueError(f"Model {model_nickname} not loaded")

        manager = self._model_managers[model_nickname]
        if not manager.is_loaded:
            raise ValueError(f"Model {model_nickname} not loaded")

        self._update_load_order(model_nickname)
        return manager

    def load_adapter(
        self,
        model_nickname: str,
        adapter_id: str,
        adapter_path: Optional[str] = None,
        lora_config: Optional[LoRAConfig] = None,
        global_id: Optional[str] = None,
        **metadata
    ) -> None:
        """Load an adapter into a specific model."""
        # Ensure model is loaded
        manager = self.get_model(model_nickname)

        # Get LoRA config if not provided
        if lora_config is None and adapter_path is None:
            model_config = self.config_loader.get_model_config(model_nickname)
            lora_config = self.config_loader.get_lora_config(model_config.family)

        # Load adapter
        manager.load_adapter(
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            lora_config=lora_config,
            **metadata
        )

        # Register globally
        global_adapter_id = global_id or f"{model_nickname}:{adapter_id}"
        self._global_adapter_registry[global_adapter_id] = (model_nickname, adapter_id)

        # Enforce adapter limits
        self._enforce_adapter_limits(model_nickname)

    def unload_adapter(
        self,
        model_nickname: str,
        adapter_id: str,
        global_id: Optional[str] = None
    ) -> None:
        """Unload an adapter from a specific model."""
        if model_nickname not in self._model_managers:
            logger.warning(f"Model {model_nickname} not found")
            return

        manager = self._model_managers[model_nickname]
        manager.unload_adapter(adapter_id)

        # Remove from global registry
        global_adapter_id = global_id or f"{model_nickname}:{adapter_id}"
        if global_adapter_id in self._global_adapter_registry:
            del self._global_adapter_registry[global_adapter_id]

    def activate_adapter(self, model_nickname: str, adapter_id: str) -> None:
        """Activate an adapter on a specific model."""
        manager = self.get_model(model_nickname)
        manager.activate_adapter(adapter_id)

    def deactivate_adapter(self, model_nickname: str) -> None:
        """Deactivate the current adapter on a specific model."""
        manager = self.get_model(model_nickname)
        manager.deactivate_adapter()

    @contextmanager
    def use_adapter(self, model_nickname: str, adapter_id: str):
        """Context manager to temporarily use an adapter."""
        manager = self.get_model(model_nickname)
        original_adapter = manager.active_adapter

        try:
            manager.activate_adapter(adapter_id)
            yield manager
        finally:
            if original_adapter is not None:
                manager.activate_adapter(original_adapter)
            else:
                manager.deactivate_adapter()

    def find_adapter(self, global_adapter_id: str) -> Optional[Tuple[str, str]]:
        """Find which model contains a specific adapter."""
        return self._global_adapter_registry.get(global_adapter_id)

    def list_all_adapters(self) -> Dict[str, List[str]]:
        """List all adapters across all models."""
        result = {}
        for nickname, manager in self._model_managers.items():
            if manager.is_loaded:
                result[nickname] = manager.loaded_adapters
        return result

    def save_adapter(
        self,
        model_nickname: str,
        adapter_id: str,
        save_path: str
    ) -> None:
        """Save an adapter to disk."""
        manager = self.get_model(model_nickname)
        manager.save_adapter(adapter_id, save_path)

    def clone_adapter(
        self,
        source_model: str,
        source_adapter: str,
        target_model: str,
        target_adapter: str,
        temp_save_path: Optional[str] = None
    ) -> None:
        """Clone an adapter from one model to another."""
        if temp_save_path is None:
            temp_save_path = f"/tmp/siera_adapter_clone_{int(time.time())}"

        # Save source adapter
        self.save_adapter(source_model, source_adapter, temp_save_path)

        # Load into target model
        self.load_adapter(target_model, target_adapter, adapter_path=temp_save_path)

        # Cleanup temp files
        import shutil
        try:
            shutil.rmtree(temp_save_path)
        except:
            logger.warning(f"Failed to cleanup temp adapter path: {temp_save_path}")

    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage for all loaded models."""
        usage = {}
        for nickname, manager in self._model_managers.items():
            if manager.is_loaded:
                usage[nickname] = manager.memory_usage
        return usage

    def get_system_memory(self) -> Dict[str, float]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        gpu_memory = {}

        # Only check GPU memory if CUDA was already initialized
        # (avoid initializing CUDA here which breaks vLLM multiprocessing)
        try:
            if torch.cuda.is_initialized():
                for i in range(torch.cuda.device_count()):
                    gpu_memory[f"cuda:{i}"] = {
                        "allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                        "reserved": torch.cuda.memory_reserved(i) / 1024**3,    # GB
                        "total": torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                    }
        except:
            # If CUDA check fails, just skip GPU memory reporting
            pass

        return {
            "ram": {
                "used": memory.used / 1024**3,  # GB
                "available": memory.available / 1024**3,  # GB
                "total": memory.total / 1024**3,  # GB
                "percent": memory.percent
            },
            "gpu": gpu_memory
        }

    def cleanup(self, max_age_seconds: int = 3600) -> None:
        """Clean up old adapters across all models."""
        for manager in self._model_managers.values():
            if manager.is_loaded:
                manager.cleanup_old_adapters(max_age_seconds)

    def _update_load_order(self, model_nickname: str) -> None:
        """Update the LRU order for models."""
        if model_nickname in self._load_order:
            self._load_order.remove(model_nickname)
        self._load_order.append(model_nickname)

    def _enforce_memory_limits(self) -> None:
        """Enforce maximum number of loaded models."""
        while len(self.loaded_models) >= self.memory_config.max_models_loaded:
            if not self._load_order:
                break

            # Unload the least recently used model
            lru_model = self._load_order[0]
            logger.info(f"Unloading LRU model {lru_model} to free memory")
            self.unload_model(lru_model)

    def _enforce_adapter_limits(self, model_nickname: str) -> None:
        """Enforce maximum number of adapters per model."""
        manager = self._model_managers[model_nickname]

        while len(manager.loaded_adapters) > self.memory_config.max_adapters_per_model:
            # Find least recently used adapter
            oldest_adapter = None
            oldest_time = float('inf')

            for adapter_id, adapter_info in manager.list_adapters().items():
                if adapter_id != manager.active_adapter and adapter_info.last_used < oldest_time:
                    oldest_time = adapter_info.last_used
                    oldest_adapter = adapter_id

            if oldest_adapter:
                logger.info(f"Unloading LRU adapter {oldest_adapter} from {model_nickname}")
                manager.unload_adapter(oldest_adapter)
            else:
                break

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all models."""
        for nickname in list(self._model_managers.keys()):
            self.unload_model(nickname)