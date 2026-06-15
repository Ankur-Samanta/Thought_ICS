# Dataset Local Cache

This directory contains locally cached versions of all datasets used in the TREE project.

## Purpose

- **Fast Loading**: Load datasets from local files instead of downloading from HuggingFace every time
- **Offline Access**: Work without internet connection once datasets are downloaded
- **Consistent Format**: All datasets are pre-processed and normalized to a common format

## Datasets

### MATH-500
- **Files**: `math500.json`, `math500_level{1-5}.json`
- **Source**: HuggingFaceH4/MATH-500
- **Size**: 500 problems total
  - Level 1: 43 problems
  - Level 2: 90 problems
  - Level 3: 105 problems
  - Level 4: 128 problems
  - Level 5: 134 problems

### GSM8K
- **Files**: `gsm8k_train.json`, `gsm8k_test.json`
- **Source**: gsm8k (main config)
- **Size**:
  - Train: 7,473 problems
  - Test: 1,319 problems

### AIME
- **Files**: `aime.json`
- **Source**: gneubig/aime-1983-2024
- **Size**: 933 problems (1983-2024)

### AMC23
- **Files**: `amc23.json`
- **Source**: math-ai/amc23
- **Size**: 40 problems

## Data Format

All datasets are stored as JSON files with the following structure:

```json
[
  {
    "problem": "Problem statement...",
    "answer": "Ground truth answer",
    "unique_id": "unique_problem_id",
    "subject": "Subject area",
    "level": 1-5
  },
  ...
]
```

## Usage

The dataset loaders in `dataset_loaders.py` automatically check for local cache first:

```python
from dataset_loaders import load_math500, load_gsm8k, load_aime, load_amc23

# Loads from local cache if available, otherwise downloads from HuggingFace
problems = load_math500(level=5, n_problems=10)
```

## Regenerating Cache

To download/update the local cache:

```bash
python download_datasets.py
```

This will:
1. Download all datasets from HuggingFace
2. Process and normalize them
3. Save to this directory

## Testing

To verify the local cache is working correctly:

```bash
python test_local_cache.py
```

This tests:
- Loading from cache works
- Data format is correct
- All required fields are present
- Fallback to HuggingFace works if cache is missing

## Notes

- Files are gitignored to avoid committing large datasets
- Download script can be re-run safely to update cache
- Original HuggingFace data format is preserved exactly
- Compatible with all existing pipelines
