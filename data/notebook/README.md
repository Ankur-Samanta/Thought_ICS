# Baseline CoT Evaluation - Jupyter Notebook

Self-contained Jupyter notebook version of `baseline_cot_eval.py` that works with any inference API.

## Quick Start

1. Open the notebook:
   ```bash
   jupyter notebook baseline_cot_eval_notebook.ipynb
   ```

2. Run all cells in order

3. To use your own API, replace the `batch_inference()` function in Cell 2:
   ```python
   def batch_inference(prompts, temperature=1.0, max_tokens=2048, **kwargs):
       # Your custom API call here
       return list_of_completions
   ```

## What's Included

- All baseline types: `single`, `iterative_l1/l2/l3/l4`, `majority_vote`
- Datasets: GSM8K, MATH-500, AMC23, AIME (loads from local cache)
- Temperature control: generation (1.0), resample (0.7), judge (0.3)
- Auto-installs dependencies
- Checkpointing and resume

## Configuration

**Cell 2** - Change model:
```python
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
```

**Cell 7** - Run evaluation:
```python
results = run_baseline_evaluation(
    baseline_type='single',
    dataset='gsm8k',
    n_problems=100,
    generation_temp=1.0,  # Initial generation (higher = more diverse)
    resample_temp=0.7,    # Correction (balanced)
    judge_temp=0.3        # Error detection (lower = more consistent)
)
```

## Requirements

```bash
pip install transformers torch datasets tqdm accelerate jupyter
```
