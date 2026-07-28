#!/usr/bin/env python3
"""
Batch Evaluation Script with Chain Caching
Evaluates on 100 level-5 MATH problems with shared initial chains across autonomy levels
不同 AutonomyLevel 可以共享相同的初始推理链
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import sys
from pathlib import Path
sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'thought_ics').is_dir())))

import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional
from datasets import load_dataset # load_dataset() 是 Hugging Face datasets 库的接口
from tqdm import tqdm # 用于显示进度条

from thought_ics.thought_mdp import (
    ToTAgent, ToTEnvironment, TreeSearch,
    initialize_model, initialize_model_3p, get_completed_paths
)
from thought_ics.self_correction import (
    identify_error_step,
    identify_error_step_incremental,
    identify_error_step_with_mv,
    generate_from_prefix,
    extract_boxed_answer,
    verify_solution_correctness
)
from thought_ics.chain_cache import save_initial_chains, load_initial_chains
from thought_ics.datasets import load_dataset_by_name, get_dataset_info, normalize_answer
# Weights & Biases，简称 WandB，是实验追踪平台
from thought_ics.utils.wandb_utils import (
    WandbConfig, init_wandb_run, log_metrics,
    log_problem_result, log_summary_metrics, finish_run,
    create_run_name
)
from thought_ics.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experiment configuration (defaults)
GENERATION_TEMP = 1.0  # Initial ToT generation (affects cache)
RESAMPLE_TEMP = 0.7    # Correction/regeneration (no cache impact)
JUDGE_TEMP = 0.3       # Error detection/verification (no cache impact)
MAX_DEPTH = 100  # Deep enough for complex problems, avoids recursion limit
MAX_TOKENS_PER_THOUGHT = None  # No limit
SEED = 42


def generate_or_load_initial_chains(
    manager,
    problems: List[Dict],
    model_name: str,
    dataset_name: str,
    temperature: float,
    max_depth: int,
    max_tokens_per_thought: int,
    n_problems: int,
    seed: int,
    model_seed: Optional[int] = None
) -> List[Dict]:
    """Generate initial chains or load from cache."""

    # Try to load from cache
    cached_chains = load_initial_chains(
        model_name=model_name,
        dataset_name=dataset_name,
        n_problems=n_problems,
        seed=seed,
        temperature=temperature,
        max_depth=max_depth,
        max_tokens_per_thought=max_tokens_per_thought,
        model_seed=model_seed
    )

    if cached_chains is not None:
        logger.info("Using cached initial chains")
        return cached_chains

    # Generate new chains
    logger.info("Generating initial chains (not found in cache)...")

    # 每处理一道题，就向其中添加一个结果字典
    initial_chains = []

    for idx, item in enumerate(tqdm(problems, desc="Generating initial chains")):
        logger.info(f"Generating initial chain for problem {idx+1}/{len(problems)}")

        # 每道题有自己的 Agent 统计和状态
        # 远程 API 客户端配置保持相同,manager相同
        agent = ToTAgent(manager, temperature=temperature, max_tokens=max_tokens_per_thought)
        env = ToTEnvironment(max_depth=max_depth)
        search = TreeSearch(agent, env, strategy="dfs", n_rollouts=1)

        root = search.search(item['problem'], verbose=False)
        completed_paths = get_completed_paths(root)

        if not completed_paths:
            logger.warning(f"No completed paths found for problem {idx+1}")
            chain = []
        else:
            chain = completed_paths[0][1:]  # Skip question

        answer = extract_boxed_answer(chain[-1] if chain else "")

        initial_chains.append({
            'problem_id': item['unique_id'],
            'problem_number': idx + 1,
            'chain': chain,
            'answer': answer,
            'correct': answer == item['answer']
        })

        logger.info(f"Problem {idx+1}: Generated chain with {len(chain)} steps, answer: {answer}, correct: {answer == item['answer']}")

    # Save to cache
    save_initial_chains(
        chains=initial_chains,
        model_name=model_name,
        dataset_name=dataset_name,
        n_problems=n_problems,
        seed=seed,
        temperature=temperature,
        max_depth=max_depth,
        max_tokens_per_thought=max_tokens_per_thought,
        model_seed=model_seed
    )

    return initial_chains

# 负责“一道题的纠错过程”
def run_iterative_correction_with_cached_chain(
    manager,
    problem: str,
    ground_truth: str,  # The correct answer for the problem
    initial_chain: List[str],
    autonomy_level: int,
    max_iterations: int,
    error_detection_method: str = 'batch',
    shared_prefix: bool = True,
    resample_temp: float = 0.7,
    judge_temp: float = 0.3,
    no_auto_stop: bool = False,
    use_context: bool = False, # 决定重新生成时，是否向模型展示上一次失败的完整推理和错误分析
    use_3p_localize: bool = False,
    api_key_3p: Optional[str] = None,
    model_3p: str = 'gpt-4o',
    base_url_3p: Optional[str] = None,
    verify: bool = False, # 决定每一轮定位错误之前，是否先问模型：你认为当前答案正确吗（self-verification）
    mv_verify: bool = False, # 多数投票认证
    mv_k: int = 5,
    mv_criterion: str = "unanimous", # 如何根据多个结果决定是否认为答案正确
    use_mv_localization: bool = False, # 多数投票错误定位
    mv_localization_k: int = 10,
    mv_localization_temp: float = 0.5
) -> Dict:
    """Run iterative correction starting from a cached initial chain.

    Args:
        error_detection_method: 'batch' (default, single-pass) or 'incremental' (step-by-step)
        shared_prefix: Whether to preserve correct prefix when regenerating (default: True)
        resample_temp: Temperature for correction/regeneration
        judge_temp: Temperature for error detection/verification
        use_3p_localize: Use 3rd-party API for error localization only
        api_key_3p: API key for 3rd-party service
        model_3p: Model to use for 3rd-party inference
        base_url_3p: Optional OpenAI-compatible API base URL
    """

    iterations = []
    chain = initial_chain
    answer = extract_boxed_answer(chain[-1] if chain else "")

    iterations.append({
        'iteration': 0, # 表示还没有纠错，是初始模型输出
        'chain': chain,
        'answer': answer,
        'correct': normalize_answer(answer) == normalize_answer(ground_truth),
        'error_step': None,
        'error_reasoning': None,
        'verify_reasoning': None,
        'model_believes_correct': None,
        'prefix_length': None,
        'localization_decisions': None
    })

    # Track previous chain for historical context (if enabled)
    previous_chain = None
    previous_error_reasoning = None

    # Iterative correction（Python 的 range 不包含结束值）
    for i in range(1, max_iterations + 1):
        # Check if we got it right
        if not no_auto_stop and normalize_answer(answer) == normalize_answer(ground_truth):
            logger.info(f"SUCCESS! Correct answer found at iteration {i-1}")
            break

        # Optional verification: ask model if it thinks answer is correct
        # Initialize verification tracking variables
        iter_verify_reasoning = None
        iter_model_believes_correct = None

        if verify:
            believes_correct, verify_reasoning = verify_solution_correctness(
                manager, problem, chain, temperature=judge_temp,
                mv_verify=mv_verify, mv_k=mv_k, mv_criterion=mv_criterion
            )
            is_actually_correct = normalize_answer(answer) == normalize_answer(ground_truth)
            logger.info(f"Verification result: model_believes_correct={believes_correct}, actually_correct={is_actually_correct}")

            # Store for inclusion in iteration data
            iter_verify_reasoning = verify_reasoning
            iter_model_believes_correct = believes_correct

            if believes_correct:
                logger.info(f"Model believes answer is correct - stopping iteration.")
                iterations.append({
                    'iteration': i,
                    'chain': chain,
                    'answer': answer,
                    'correct': is_actually_correct,
                    'error_step': None,
                    'error_reasoning': None,
                    'verify_reasoning': verify_reasoning,
                    'model_believes_correct': True,
                    'prefix_length': None,
                    'localization_decisions': None
                })
                break
            else:
                logger.info(f"Model believes answer is incorrect - continuing to error detection.")

        # Identify error step using selected method
        all_localization_decisions = None  # Track all MV decisions if enabled

        if use_mv_localization:
            # Use majority vote localization
            error_step, error_reasoning, all_localization_decisions = identify_error_step_with_mv(
                manager, problem, chain, ground_truth, autonomy_level,
                temperature=mv_localization_temp, mv_k=mv_localization_k
            )
        elif use_3p_localize:
            # Use 3rd-party API for error localization
            from thought_ics.localization.third_party_api import call_3p_error_localization
            error_step, error_reasoning = call_3p_error_localization(
                problem, chain, ground_truth, autonomy_level,
                method=error_detection_method, api_key=api_key_3p, model=model_3p,
                base_url=base_url_3p
            )
        elif error_detection_method == 'incremental':
            error_step, error_reasoning = identify_error_step_incremental(manager, problem, chain, ground_truth, autonomy_level, temperature=judge_temp)
        else:  # default: 'batch'
            error_step, error_reasoning = identify_error_step(manager, problem, chain, ground_truth, autonomy_level, temperature=judge_temp)

        # Check if model found no errors
        if error_step == 0:
            is_correct = normalize_answer(answer) == normalize_answer(ground_truth)
            logger.info(f"Model found no errors - stopping iteration. Answer correct: {is_correct}")
            iterations.append({
                'iteration': i,
                'chain': chain,
                'answer': answer,
                'correct': is_correct,
                'error_step': 0,
                'error_reasoning': error_reasoning,
                'verify_reasoning': iter_verify_reasoning,
                'model_believes_correct': iter_model_believes_correct,
                'prefix_length': None,
                'localization_decisions': all_localization_decisions
            })
            break

        # Generate new chain from before error
        if shared_prefix:
            prefix = chain[:error_step - 1]  # Steps before the error
            logger.info(f"Backtracking to step {error_step-1}, keeping {len(prefix)} steps as prefix")
        else:
            prefix = []  # Force full regeneration from scratch
            logger.info(f"Error at step {error_step}, regenerating entire solution from scratch (no shared prefix)")

        # Store the chain we're moving away from (if historical context enabled)
        if use_context:
            previous_chain = chain
            previous_error_reasoning = error_reasoning

        # Regenerate (with historical context if enabled)
        if use_context and previous_chain is not None:
            chain = generate_from_prefix(manager, problem, prefix,
                                        previous_chain=previous_chain,
                                        error_reasoning=previous_error_reasoning,
                                        error_step=error_step,
                                        temperature=resample_temp)
        else:
            chain = generate_from_prefix(manager, problem, prefix, temperature=resample_temp)

        answer = extract_boxed_answer(chain[-1] if chain else "")

        iterations.append({
            'iteration': i,
            'chain': chain,
            'answer': answer,
            'correct': normalize_answer(answer) == normalize_answer(ground_truth),
            'error_step': error_step,
            'error_reasoning': error_reasoning,
            'verify_reasoning': iter_verify_reasoning,
            'model_believes_correct': iter_model_believes_correct,
            'prefix_length': len(prefix),
            'localization_decisions': all_localization_decisions
        })

        logger.info(f"Iteration {i}: Answer = {answer}, Correct = {normalize_answer(answer) == normalize_answer(ground_truth)}")

        if not no_auto_stop and normalize_answer(answer) == normalize_answer(ground_truth):
            logger.info(f"SUCCESS! Correct answer found at iteration {i}")
            break

    final_correct = iterations[-1]['correct']

    return {
        'problem': problem,
        'ground_truth': ground_truth,
        'iterations': iterations,
        'success': final_correct,
        'total_iterations': len(iterations)
    }

# 负责“整批题目的实验管理”
def run_batch_evaluation(  
    autonomy_level: int,
    gpu_ids: str,
    tensor_parallel_size: int,
    n_problems: int = 100,
    max_iterations: int = 10,
    dataset: str = "math500",
    level: Optional[int] = None,  # 主要用于按照难度筛选题目
    model_name: str = "llama8b",
    experiment_name: Optional[str] = None,
    wandb_project: str = "thought-ics",
    wandb_entity: Optional[str] = None,
    enable_wandb: bool = False,
    error_detection_method: str = 'batch',
    shared_prefix: bool = True,
    generation_temp: float = 1.0,
    resample_temp: float = 0.7,
    judge_temp: float = 0.3,
    seed: int = 42,
    model_seed: Optional[int] = None,
    no_auto_stop: bool = False,
    use_context: bool = False,
    use_3p: bool = False,
    use_3p_localize: bool = False,
    api_key_3p: Optional[str] = None,
    model_3p: str = 'gpt-4o',
    base_url_3p: Optional[str] = None,
    manager=None,  # Optional pre-configured manager (for notebook use)
    verify: bool = False,
    mv_verify: bool = False,
    mv_k: int = 5,
    mv_criterion: str = "unanimous",
    use_mv_localization: bool = False,
    mv_localization_k: int = 10,
    mv_localization_temp: float = 0.5
):
    """Run batch evaluation on problems with cached initial chains.

    Args:
        autonomy_level: 1-3 (Oracle, Binary, Autonomous); historical-context conditioning is
            orthogonal and set via use_context
        gpu_ids: GPU IDs to use (ignored if use_3p=True)
        tensor_parallel_size: Tensor parallel size (ignored if use_3p=True)
        n_problems: Number of problems to evaluate
        max_iterations: Max correction iterations per problem
        dataset: Dataset name ("math500", "gsm8k", or "amc23")
        level: For MATH-500, filter by difficulty level (1-5)
        shared_prefix: Whether to preserve correct prefix when regenerating (default: True)
        generation_temp: Temperature for initial ToT generation (affects cache)
        resample_temp: Temperature for correction/regeneration (no cache impact)
        judge_temp: Temperature for error detection/verification (no cache impact)
        model_name: Model nickname (default: "llama8b")
        seed: Random seed for dataset sampling (default: 42)
        model_seed: Seed for model generation (None=non-deterministic, int=deterministic)
        experiment_name: Optional custom experiment name
        wandb_project: Wandb project name
        wandb_entity: Wandb entity (username or team)
        enable_wandb: Whether to enable wandb logging
        error_detection_method: 'batch' (default, single-pass) or 'incremental' (step-by-step)
        use_3p: Use 3rd-party API for ALL inference (no local vLLM)
        use_3p_localize: Use 3rd-party API for error localization only
        api_key_3p: API key for 3rd-party service
        model_3p: Model to use for 3rd-party inference (default: gpt-4o)
        base_url_3p: Optional OpenAI-compatible API base URL
        manager: Optional pre-configured model manager (for notebook use). If provided,
            skips model initialization and uses this manager directly.
    """

    autonomy_names = {1: "L1_Oracle", 2: "L2_Binary", 3: "L3_Autonomous"}
    autonomy_name = autonomy_names.get(autonomy_level, f"L{autonomy_level}")

    # Get dataset info
    dataset_info = get_dataset_info(dataset)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name is None:
        level_suffix = f"_L{level}" if level is not None else ""
        prefix_suffix = "_no_shared_prefix" if not shared_prefix else ""
        # Add context suffix if historical-context conditioning is enabled (any level)
        context_suffix = "_with_context" if use_context else ""
        # Add 3P suffix if using 3P (OpenAI-compatible API) mode
        if use_3p:
            model_suffix = f"_3p_{model_3p}"
        else:
            model_suffix = f"_{model_name}"
        experiment_name = f"eval{model_suffix}_{autonomy_name}_{dataset}{level_suffix}{prefix_suffix}{context_suffix}_{timestamp}"

    exp_dir = Path("experiments") / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to experiment directory
    log_file = exp_dir / "run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Logging to: {log_file}")

    # Save experiment config
    config = {
        'experiment_name': experiment_name,
        'autonomy_level': autonomy_level,
        'autonomy_name': autonomy_name,
        # Model configuration
        'model_name': model_name,  # Local model name (may be ignored if use_3p=True)
        'model_seed': model_seed,
        'gpu_ids': gpu_ids,
        'tensor_parallel_size': tensor_parallel_size,
        # 3P configuration
        'use_3p': use_3p,
        'use_3p_localize': use_3p_localize,
        'model_3p': model_3p,
        'inference_backend': 'openai_api' if use_3p else 'vllm',
        # Evaluation settings
        'n_problems': n_problems,
        'max_iterations': max_iterations,
        'generation_temp': generation_temp,
        'resample_temp': resample_temp,
        'judge_temp': judge_temp,
        'max_depth': MAX_DEPTH,
        'max_tokens_per_thought': MAX_TOKENS_PER_THOUGHT,
        'dataset': dataset,
        'dataset_info': dataset_info,
        'level_filter': level,
        'seed': seed,
        'error_detection_method': error_detection_method,
        'shared_prefix': shared_prefix,
        'no_auto_stop': no_auto_stop,
        'use_context': use_context,
        # MV localization settings
        'use_mv_localization': use_mv_localization,
        'mv_localization_k': mv_localization_k,
        'mv_localization_temp': mv_localization_temp,
        'timestamp': datetime.now().isoformat()
    }

    config_file = exp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize wandb
    wandb_run = None
    if enable_wandb:
        wandb_config = WandbConfig(
            enabled=True,
            project=wandb_project,
            entity=wandb_entity,
            tags=["batch_eval", f"autonomy_l{autonomy_level}", model_name, dataset],
            notes=f"Batch evaluation with {autonomy_name} on {dataset}"
        )

        run_name = create_run_name(
            model_name=model_name,
            dataset_name=dataset,
            experiment_type="tot_eval",
            autonomy=f"L{autonomy_level}",
            level=f"lvl{level}" if level else None
        )

        wandb_run = init_wandb_run(
            config=wandb_config,
            run_name=run_name,
            job_type="evaluation",
            run_config=config
        )
        logger.info(f"Wandb run initialized: {run_name}")

    logger.info("="*100)
    logger.info(f"BATCH EVALUATION WITH CACHING - {autonomy_name}")
    logger.info("="*100)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Dataset: {dataset_info['name']}")
    if level is not None:
        logger.info(f"Level filter: {level}")
    logger.info(f"Output dir: {exp_dir}")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Tensor Parallel Size: {tensor_parallel_size}")
    logger.info(f"Number of problems: {n_problems}")
    logger.info(f"Max iterations per problem: {max_iterations}")
    logger.info(f"Error detection method: {error_detection_method}")
    logger.info(f"Generation temperature: {generation_temp}")
    logger.info(f"Resample temperature: {resample_temp}")
    logger.info(f"Judge temperature: {judge_temp}")
    logger.info(f"Max depth: {MAX_DEPTH}")
    logger.info(f"Max tokens per thought: {MAX_TOKENS_PER_THOUGHT}")
    logger.info("="*100)

    # Load problems
    logger.info(f"Loading {dataset} dataset...")
    problems = load_dataset_by_name(
        dataset_name=dataset,
        n_problems=n_problems,
        level=level,
        seed=seed
    )

    # Initialize model (local vLLM or OpenAI-compatible API)
    # If manager is pre-configured (notebook use), skip initialization
    if manager is not None:
        logger.info("Using pre-configured manager (notebook mode)")
        if use_3p:
            effective_model_name = model_3p
        else:
            effective_model_name = model_name
    elif use_3p:
        logger.info(f"Initializing 3P API model '{model_3p}' (no local GPU required)...")
        manager = initialize_model_3p(
            api_key=api_key_3p,
            model=model_3p,
            base_url=base_url_3p,
        )
        effective_model_name = model_3p
    else:
        logger.info(f"Initializing local model '{model_name}' on GPUs {gpu_ids}...")
        manager = initialize_model(gpu_ids=gpu_ids, tensor_parallel_size=tensor_parallel_size, model_name=model_name, model_seed=model_seed)
        effective_model_name = model_name

    # Generate or load initial chains
    initial_chains = generate_or_load_initial_chains(
        manager=manager,
        problems=problems,
        model_name=effective_model_name,
        dataset_name=dataset,
        temperature=generation_temp,
        max_depth=MAX_DEPTH,
        max_tokens_per_thought=MAX_TOKENS_PER_THOUGHT,
        n_problems=n_problems,
        seed=seed,
        model_seed=model_seed
    )

    # Check for existing progress (resume capability)
    checkpoint_file = exp_dir / "checkpoint.json"
    completed_problem_ids = set()
    results = []

    if checkpoint_file.exists():
        logger.info(f"Found existing checkpoint: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                completed_problem_ids = set(r['problem_id'] for r in results)
                logger.info(f"Resuming from checkpoint: {len(results)}/{len(problems)} problems already completed")
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}. Starting fresh.")
            results = []
            completed_problem_ids = set()

    # Run evaluation with cached chains
    stats = {
        'total_problems': len(problems),
        'successful': sum(1 for r in results if r.get('success', False)),
        'failed': sum(1 for r in results if not r.get('success', False)),
        'total_iterations': sum(r.get('total_iterations', 0) for r in results),
        'avg_iterations': 0.0,
        'config': config
    }

    logger.info("\nStarting evaluation with cached initial chains...")
    for idx, (item, init_chain_data) in enumerate(zip(problems, initial_chains)):
        problem_num = idx + 1

        # Skip if already completed (resume capability)
        if item['unique_id'] in completed_problem_ids:
            logger.info(f"Skipping problem {problem_num}/{len(problems)} (already completed)")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"Problem {problem_num}/{len(problems)}")
        logger.info(f"Subject: {item['subject']}, Level: {item['level']}")
        logger.info(f"Initial chain: {len(init_chain_data['chain'])} steps, answer: {init_chain_data['answer']}, correct: {init_chain_data['correct']}")
        logger.info(f"{'='*80}")

        # Full API mode already routes localization through the API-backed manager,
        # preserving L1/L2/L3 prompt semantics. This special path is only for a
        # local generation model paired with an external L2 localizer.
        effective_use_3p_localize = use_3p_localize

        try:
            result = run_iterative_correction_with_cached_chain(
                manager=manager,
                problem=item['problem'],
                ground_truth=item['answer'],
                initial_chain=init_chain_data['chain'],
                autonomy_level=autonomy_level,
                max_iterations=max_iterations,
                error_detection_method=error_detection_method,
                shared_prefix=shared_prefix,
                resample_temp=resample_temp,
                judge_temp=judge_temp,
                no_auto_stop=no_auto_stop,
                use_context=use_context,
                use_3p_localize=effective_use_3p_localize,
                api_key_3p=api_key_3p,
                model_3p=model_3p,
                base_url_3p=base_url_3p,
                verify=verify,
                mv_verify=mv_verify,
                mv_k=mv_k,
                mv_criterion=mv_criterion,
                use_mv_localization=use_mv_localization,
                mv_localization_k=mv_localization_k,
                mv_localization_temp=mv_localization_temp
            )

            result['problem_id'] = item['unique_id']
            result['subject'] = item['subject']
            result['level'] = item['level']
            result['problem_number'] = problem_num

            results.append(result)

            if result['success']:
                stats['successful'] += 1
            else:
                stats['failed'] += 1

            stats['total_iterations'] += result['total_iterations']

            logger.info(f"Result: {'SUCCESS' if result['success'] else 'FAILED'} "
                       f"(iterations: {result['total_iterations']})")

            # Log to wandb
            if enable_wandb and wandb_run is not None:
                log_problem_result(
                    problem_id=item['unique_id'],
                    problem_number=problem_num,
                    predicted_answer=result['iterations'][-1]['answer'],
                    ground_truth=item['answer'],
                    correct=result['success'],
                    iterations=result['total_iterations'],
                    additional_metrics={
                        'subject': item['subject'],
                        'level': item['level'],
                        'autonomy_level': autonomy_level
                    }
                )

            # Save checkpoint after each problem
            with open(checkpoint_file, 'w') as f:
                json.dump({'results': results}, f, indent=2)

        except Exception as e:
            logger.error(f"Error on problem {problem_num}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'problem_id': item['unique_id'],
                'problem': item['problem'],
                'ground_truth': item['answer'],
                'subject': item['subject'],
                'level': item['level'],
                'problem_number': problem_num,
                'error': str(e),
                'success': False
            })
            stats['failed'] += 1

            # Save checkpoint even after errors
            with open(checkpoint_file, 'w') as f:
                json.dump({'results': results}, f, indent=2)

    # Compute final stats
    stats['avg_iterations'] = stats['total_iterations'] / stats['total_problems'] if stats['total_problems'] > 0 else 0
    stats['success_rate'] = (stats['successful'] / stats['total_problems']) * 100 if stats['total_problems'] > 0 else 0

    # Save results
    results_file = exp_dir / "results.json"
    full_results = {
        'stats': stats,
        'results': results
    }

    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"\n{'='*100}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*100}")
    logger.info(f"Total problems: {stats['total_problems']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Success rate: {stats['success_rate']:.2f}%")
    logger.info(f"Average iterations: {stats['avg_iterations']:.2f}")
    logger.info(f"Total iterations: {stats['total_iterations']}")
    logger.info(f"\nResults saved to: {exp_dir}")
    logger.info(f"  - Config: {config_file}")
    logger.info(f"  - Results: {results_file}")
    logger.info(f"{'='*100}")

    # Log summary to wandb
    if enable_wandb and wandb_run is not None:
        log_summary_metrics(
            total_problems=stats['total_problems'],
            correct_problems=stats['successful'],
            mean_iterations=stats['avg_iterations'],
            additional_summary={
                'success_rate': stats['success_rate'],
                'total_iterations': stats['total_iterations'],
                'autonomy_level': autonomy_level,
                'dataset': dataset,
                'model': model_name
            }
        )
        finish_run()

    # Compute comprehensive metrics
    logger.info(f"\n{'='*100}")
    logger.info("Computing comprehensive metrics...")
    logger.info(f"{'='*100}")
    try:
        metrics = compute_metrics(exp_dir)
        metrics_file = exp_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Metrics saved to: {metrics_file}")

        # Log key metrics
        overall = metrics.get("overall_performance", {})
        logger.info(f"  First Attempt Accuracy: {overall.get('first_attempt_accuracy', 0):.1%}")
        logger.info(f"  Final Accuracy: {overall.get('final_accuracy', 0):.1%}")
        logger.info(f"  Improvement: {overall.get('absolute_improvement', 0):.1%}")

        # Log C1 error detection if available
        if "error_detection_ability" in metrics:
            ed = metrics["error_detection_ability"]
            pm = ed.get("performance_metrics", {})
            logger.info(f"  Error Detection - Precision: {pm.get('precision', 0):.1%}, Recall: {pm.get('recall', 0):.1%}")
    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}")
        logger.error("Continuing anyway...")

    # Cleanup
    manager.unload_base_model()

    return full_results


def main():
    parser = argparse.ArgumentParser(description='Batch Evaluation with Chain Caching')
    parser.add_argument('--autonomy-level', type=int, choices=[1, 2, 3], required=True,
                        help='Autonomy level: 1=Oracle, 2=Binary Feedback, 3=Full Autonomy. '
                             'Historical-context conditioning is orthogonal — add --context at any level.')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs (e.g., "0,1"). Required for local inference; ignored with --3p.')
    parser.add_argument('--tensor-parallel-size', type=int, default=None,
                        help='Number of GPUs for tensor parallelism. Required for local inference; ignored with --3p.')
    parser.add_argument('--dataset', type=str, default='math500',
                        choices=['math500', 'gsm8k', 'amc23', 'aime', 'csqa', 'gpqa', 'svamp', 'mathqa', 'imo', 'imobench'],
                        help='Dataset to evaluate on (default: math500)')
    parser.add_argument('--level', type=int, default=None,
                        help='For MATH-500, filter by difficulty level 1-5 (default: all levels)')
    parser.add_argument('--model', type=str, default='llama8b',
                        choices=['llama8b', 'llama70b', 'qwen7b', 'qwen14b', 'qwen32b', 'qwen2b', 'llama3b', 'phi4b',
                                 'mistral3b', 'mistral8b', 'mistral14b', 'gptoss20b', 'gptoss120b'],
                        help='Model to use (default: llama8b)')
    parser.add_argument('--n-problems', type=int, default=100,
                        help='Number of problems to evaluate (default: 100)')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum correction iterations per problem (default: 10)')
    parser.add_argument('--error-detection', type=str, choices=['batch', 'incremental'], default='batch',
                        help='Error detection method: batch=single-pass (default), incremental=step-by-step verification')
    parser.add_argument('--no-shared-prefix', action='store_true',
                        help='Disable shared prefix - regenerate entire solution from scratch instead of preserving correct steps')
    parser.add_argument('--no-auto-stop', action='store_true',
                        help='Disable auto-stopping on correct answer - always run max_iterations (default: False)')
    parser.add_argument('--context', action='store_true',
                        help='Condition the resample on the prior iteration'"'"'s failed chain and error '
                             'analysis (instead of a fresh stochastic sample). Combinable with any --autonomy-level.')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Optional experiment name (default: auto-generated)')
    parser.add_argument('--wandb-project', type=str, default='thought-ics',
                        help='Wandb project name (default: SIERA)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity (username or team) (default: your logged-in wandb entity)')
    parser.add_argument('--enable-wandb', action='store_true',
                        help='Enable wandb logging (default: False)')
    parser.add_argument('--generation-temp', type=float, default=1.0,
                        help='Temperature for initial ToT generation (affects cache) (default: 1.0)')
    parser.add_argument('--resample-temp', type=float, default=0.7,
                        help='Temperature for correction/regeneration (no cache impact) (default: 0.7)')
    parser.add_argument('--judge-temp', type=float, default=0.3,
                        help='Temperature for error detection/verification (no cache impact) (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for dataset sampling and reproducibility (default: 42)')
    parser.add_argument('--model-seed', type=int, default=None,
                        help='Seed for model generation (None=non-deterministic, int=deterministic, default: None)')
    parser.add_argument('--3p', action='store_true',
                        help='Use 3rd-party API for ALL inference (ToT generation, error detection, correction). Skips local vLLM loading.')
    parser.add_argument('--3p-localize', action='store_true',
                        help='Use 3rd-party API for error localization only (L2). Local vLLM used for generation.')
    parser.add_argument('--3p-api-key', type=str, default=None,
                        help='API key for 3rd-party service (default: OPENAI_API_KEY env var)')
    parser.add_argument('--3p-base-url', type=str, default=None,
                        help='OpenAI-compatible API base URL (default: OPENAI_BASE_URL env var; '
                             'omit for the official OpenAI endpoint)')
    parser.add_argument('--3p-model', type=str, default=None,
                        help='Model to use for 3rd-party inference '
                             '(default: OPENAI_MODEL env var, then gpt-4o)')
    parser.add_argument('--verify', action='store_true',
                        help='Enable solution verification before error detection (L3 only)')
    parser.add_argument('--mv', action='store_true',
                        help='Enable majority vote verification (requires --verify)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of rollouts for majority vote verification (default: 5)')
    parser.add_argument('--mv-criterion', type=str, default='unanimous',
                        choices=['unanimous', 'majority', 'any'],
                        help='MV voting criterion: unanimous (all YES), majority (>50%% YES), any (>=1 YES). Default: unanimous')
    parser.add_argument('--mv-localization', action='store_true',
                        help='Use majority vote for error localization (generates K samples and takes majority vote)')
    parser.add_argument('--mv-localization-k', type=int, default=10,
                        help='Number of samples for MV localization (default: 10)')
    parser.add_argument('--mv-localization-temp', type=float, default=0.5,
                        help='Temperature for MV localization samples (default: 0.5)')

    args = parser.parse_args()

    # Validation: --mv requires --verify
    if args.mv and not args.verify:
        raise ValueError("--mv requires --verify to be enabled")

    # Historical-context conditioning is orthogonal to the oracle level: it lives on --context
    # and combines with any autonomy level (1/2/3).
    use_context = args.context

    # Get 3P flags
    use_3p = getattr(args, '3p', False)
    use_3p_localize = getattr(args, '3p_localize', False)

    # Get 3P API key from args or environment
    api_key_3p = getattr(args, '3p_api_key', None) or os.environ.get('OPENAI_API_KEY')
    base_url_3p = getattr(args, '3p_base_url', None) or os.environ.get('OPENAI_BASE_URL')
    model_3p = getattr(args, '3p_model', None) or os.environ.get('OPENAI_MODEL') or 'gpt-4o'

    # Validation: --3p requires API key
    if use_3p and api_key_3p is None:
        raise ValueError("--3p requires API key via --3p-api-key or OPENAI_API_KEY environment variable")

    # Validation: local inference requires GPU configuration (ignored under --3p)
    if use_3p:
        logger.info(f"3P mode enabled: ALL inference will use {model_3p} via OpenAI-compatible API")
        if base_url_3p:
            logger.info("Using a custom OpenAI-compatible API endpoint")
        logger.info("Local vLLM model will NOT be loaded (--gpus and --tensor-parallel-size ignored)")
    else:
        if args.gpus is None or args.tensor_parallel_size is None:
            parser.error("--gpus and --tensor-parallel-size are required for local inference "
                         "(omit them only when using --3p)")

    run_batch_evaluation(
        autonomy_level=args.autonomy_level,
        gpu_ids=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        n_problems=args.n_problems,
        max_iterations=args.max_iterations,
        dataset=args.dataset,
        level=args.level,
        model_name=args.model,
        experiment_name=args.experiment_name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        enable_wandb=args.enable_wandb,
        generation_temp=args.generation_temp,
        resample_temp=args.resample_temp,
        judge_temp=args.judge_temp,
        seed=args.seed,
        model_seed=args.model_seed,
        error_detection_method=args.error_detection,
        shared_prefix=not args.no_shared_prefix,
        no_auto_stop=args.no_auto_stop,
        use_context=use_context,
        use_3p=use_3p,
        use_3p_localize=use_3p_localize,
        api_key_3p=api_key_3p,
        model_3p=model_3p,
        base_url_3p=base_url_3p,
        verify=args.verify,
        mv_verify=args.mv,
        mv_k=args.k,
        mv_criterion=args.mv_criterion,
        use_mv_localization=args.mv_localization,
        mv_localization_k=args.mv_localization_k,
        mv_localization_temp=args.mv_localization_temp
    )


if __name__ == "__main__":
    main()
