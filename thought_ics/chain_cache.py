#!/usr/bin/env python3
"""
Chain caching system for initial ToT chain generation.
Caches chains based on model name, dataset, n_problems, seed, temperature, and other config.
"""

import os
import json
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_key(
    model_name: str,
    dataset_name: str,
    n_problems: int,
    seed: int,
    temperature: float,
    max_depth: int,
    max_tokens_per_thought: int,
    model_seed: Optional[int] = None
) -> str:
    """Generate cache key from config parameters including model name.

    Note: model_seed is only included in key if explicitly set (not None).
    This preserves backward compatibility with existing caches.
    """
    config_str = f"{model_name}_{dataset_name}_{n_problems}_{seed}_{temperature}_{max_depth}_{max_tokens_per_thought}"

    # Only append model_seed if explicitly set (backward compatibility)
    if model_seed is not None:
        config_str += f"_mseed{model_seed}"

    # Use hash for shorter filenames
    return hashlib.md5(config_str.encode()).hexdigest()


def get_cache_path(cache_key: str, cache_type: str = "tot") -> Path:
    """Get cache file path for a given cache key.

    Args:
        cache_key: The hash-based cache key
        cache_type: Either "tot" for ToT chains or "cot" for CoT chains
    """
    if cache_type == "cot":
        return CACHE_DIR / f"initial_cot_{cache_key}.json"
    else:
        return CACHE_DIR / f"initial_chains_{cache_key}.json"


def save_initial_chains(
    chains: List[Dict],
    model_name: str,
    dataset_name: str,
    n_problems: int,
    seed: int,
    temperature: float,
    max_depth: int,
    max_tokens_per_thought: int,
    model_seed: Optional[int] = None,
    cache_type: str = "tot"
) -> None:
    """Save initial chains to cache with metadata.

    Args:
        cache_type: Either "tot" for ToT chains or "cot" for CoT chains
    """
    cache_key = get_cache_key(model_name, dataset_name, n_problems, seed, temperature, max_depth, max_tokens_per_thought, model_seed)
    cache_path = get_cache_path(cache_key, cache_type)

    cache_data = {
        'metadata': {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'n_problems': n_problems,
            'seed': seed,
            'temperature': temperature,
            'max_depth': max_depth,
            'max_tokens_per_thought': max_tokens_per_thought,
            'model_seed': model_seed,
            'cache_key': cache_key
        },
        'chains': chains
    }

    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"Saved {len(chains)} initial chains to cache: {cache_path.name}")


def load_initial_chains(
    model_name: str,
    dataset_name: str,
    n_problems: int,
    seed: int,
    temperature: float,
    max_depth: int,
    max_tokens_per_thought: int,
    model_seed: Optional[int] = None,
    cache_type: str = "tot"
) -> Optional[List[Dict]]:
    """Load initial chains from cache if they exist.

    Args:
        cache_type: Either "tot" for ToT chains or "cot" for CoT chains
    """
    cache_key = get_cache_key(model_name, dataset_name, n_problems, seed, temperature, max_depth, max_tokens_per_thought, model_seed)
    cache_path = get_cache_path(cache_key, cache_type)

    if not cache_path.exists():
        print(f"Cache not found for key {cache_key}")
        return None

    with open(cache_path, 'r') as f:
        cache_data = json.load(f)

    # Verify metadata matches
    metadata = cache_data['metadata']
    if (metadata.get('model_name') == model_name and
        metadata['dataset_name'] == dataset_name and
        metadata['n_problems'] == n_problems and
        metadata['seed'] == seed and
        metadata['temperature'] == temperature and
        metadata['max_depth'] == max_depth and
        metadata['max_tokens_per_thought'] == max_tokens_per_thought and
        metadata.get('model_seed') == model_seed):

        chains = cache_data['chains']
        empty_chain_ids = [
            chain.get('problem_id', f'index_{idx}')
            for idx, chain in enumerate(chains)
            if not chain.get('chain')
        ]
        if empty_chain_ids:
            print(
                "Ignoring invalid cache containing empty reasoning chains: "
                f"{cache_path.name} ({', '.join(empty_chain_ids[:5])})"
            )
            return None

        print(f"Loaded {len(chains)} initial chains from cache: {cache_path.name}")
        return chains
    else:
        print(f"Cache metadata mismatch for key {cache_key}")
        return None


def list_cached_chains() -> List[Dict]:
    """List all cached chain files with their metadata."""
    cached = []
    for cache_file in CACHE_DIR.glob("initial_chains_*.json"):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            cached.append({
                'file': cache_file.name,
                'metadata': cache_data['metadata'],
                'num_chains': len(cache_data['chains'])
            })
    return cached
