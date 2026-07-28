#!/usr/bin/env python3
"""
Unified dataset loaders for math reasoning and multiple choice datasets.
用于载入实验需要的数据集（先从本地找，没有的话就去hugging face）

Supports:
- MATH-500: HuggingFace H4 MATH-500 (500 test problems, level 1-5)
- GSM8K: Grade School Math 8K (1.3K test problems)
- AMC23: AMC 2023 competition (40 problems)
- AIME: AIME competition problems (1983-2024)
- GPQA: Graduate-level science MCQ (4 options A-D)
- CSQA: CommonsenseQA (5 options A-E)
- MathQA: Math word problems MCQ
- SVAMP: Grade school math (numeric answers)
- IMO: IMO Shortlist problems (~80 problems from IMO-Bench)
- IMO-Bench: Full IMO-Bench AnswerBench dataset (~270 problems)

All datasets are normalized to a common format with fields:
- problem: problem statement (str) - includes MC options and instruction if applicable
- answer: ground truth answer (str) - letter for MCQ, numeric for math
- unique_id: unique problem identifier (str)
- subject: subject area (str)
- level: difficulty level (int, 1-5)

Multiple Choice:
- MC questions have options and instruction embedded in the 'problem' field
- MC answers use normalize_answer() for case-insensitive comparison

Local Cache:
- Datasets are first loaded from local cache (data/ directory) if available
- Falls back to HuggingFace if local cache doesn't exist
- Use download_datasets.py to populate local cache
"""

import csv
import io
import json
import logging
import random
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Local data directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract numerical answer from GSM8K answer format.

    GSM8K answers are in format: "reasoning text\n#### 42"
    We extract just the final number after ####

    Args:
        answer_text: Full answer text from GSM8K

    Returns:
        Extracted numerical answer as string
    """
    if '####' in answer_text:
        # Split on #### and take the last part
        parts = answer_text.split('####')
        if len(parts) >= 2:
            # Clean up the number (remove commas, whitespace, etc)
            answer = parts[-1].strip()
            # Remove commas from numbers
            answer = answer.replace(',', '')
            return answer

    # Fallback: try to find last number in the text
    numbers = re.findall(r'-?\d+\.?\d*', answer_text)
    if numbers:
        return numbers[-1].replace(',', '')

    return answer_text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (case-insensitive, whitespace stripped).

    This is particularly important for multiple choice questions where
    'a' should match 'A', ' B ' should match 'B', etc.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer (lowercase, stripped, extra spaces removed)
    """
    if not answer:
        return ""

    # Convert to lowercase
    answer = answer.lower()

    # Strip leading/trailing whitespace
    answer = answer.strip()

    # Replace multiple spaces with single space
    answer = re.sub(r'\s+', ' ', answer)

    return answer


def load_math500(
    n_problems: Optional[int] = None,
    level: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load MATH-500 dataset from local cache or HuggingFace.

    Dataset: HuggingFaceH4/MATH-500
    Size: 500 test problems across 7 subjects
    Levels: 1 (easiest) to 5 (hardest)

    Args:
        n_problems: Number of problems to return (None = all)
        level: Filter by difficulty level (None = all levels)
        seed: Random seed for sampling (only used if n_problems specified)

    Returns:
        List of problem dictionaries with keys:
        - problem: problem statement
        - answer: ground truth answer (extracted from \\boxed{})
        - unique_id: unique problem ID
        - subject: subject area
        - level: difficulty level (1-5)
    """
    # Try to load from local cache first
    if level is not None:
        local_file = DATA_DIR / f"math500_level{level}.json"
    else:
        local_file = DATA_DIR / "math500.json"

    if local_file.exists():
        logger.info(f"Loading MATH-500 from local cache: {local_file.name}")
        with open(local_file, 'r') as f:
            problems = json.load(f)
        logger.info(f"Loaded {len(problems)} MATH-500 problems from cache")
    else:
        logger.info("Loading MATH-500 dataset from HuggingFace...")
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

        problems = []
        for idx, item in enumerate(dataset):
            # Filter by level if specified
            if level is not None and item['level'] != level:
                continue

            problems.append({
                'problem': item['problem'],
                'answer': item['answer'],  # Already in boxed format
                'unique_id': item.get('unique_id', f"math500_{idx}"),
                'subject': item['subject'],
                'level': item['level']
            })

        logger.info(f"Loaded {len(problems)} MATH-500 problems from HuggingFace")

    # Sample if requested
    if n_problems is not None and n_problems < len(problems):
        import random
        random.seed(seed)
        problems = random.sample(problems, n_problems)
        logger.info(f"Sampled {len(problems)} problems (seed={seed})")

    return problems


def load_gsm8k(
    n_problems: Optional[int] = None,
    split: str = "test",
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load GSM8K dataset from local cache or HuggingFace.

    Dataset: gsm8k (main config)
    Size: 1,319 test problems, 7,473 train problems
    Levels: All grade school level (assigned level 2 for consistency)

    Args:
        n_problems: Number of problems to return (None = all)
        split: Dataset split ("test" or "train")
        seed: Random seed for sampling (only used if n_problems specified)

    Returns:
        List of problem dictionaries with keys:
        - problem: problem statement
        - answer: ground truth answer (numerical)
        - unique_id: unique problem ID
        - subject: "Math" (generic)
        - level: 2 (grade school level)
    """
    # Try to load from local cache first
    local_file = DATA_DIR / f"gsm8k_{split}.json"

    if local_file.exists():
        logger.info(f"Loading GSM8K from local cache: {local_file.name}")
        with open(local_file, 'r') as f:
            problems = json.load(f)
        logger.info(f"Loaded {len(problems)} GSM8K problems from cache")
    else:
        logger.info(f"Loading GSM8K dataset from HuggingFace (split={split})...")
        dataset = load_dataset("gsm8k", "main", split=split)

        problems = []
        for idx, item in enumerate(dataset):
            # Extract clean numerical answer from GSM8K format
            clean_answer = extract_gsm8k_answer(item['answer'])

            problems.append({
                'problem': item['question'],
                'answer': clean_answer,
                'unique_id': f"gsm8k_{split}_{idx}",
                'subject': 'Math',  # GSM8K doesn't have subject categories
                'level': 2  # Grade school level - assign level 2 (elementary)
            })

        logger.info(f"Loaded {len(problems)} GSM8K problems from {split} split")

    # Sample if requested
    if n_problems is not None and n_problems < len(problems):
        import random
        random.seed(seed)
        problems = random.sample(problems, n_problems)
        logger.info(f"Sampled {len(problems)} problems (seed={seed})")

    return problems


def load_amc23(
    n_problems: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load AMC23 dataset from local cache or HuggingFace.

    Dataset: math-ai/amc23
    Size: 40 problems from 2023 AMC competition
    Levels: All competition level (assigned level 3 for consistency)

    Args:
        n_problems: Number of problems to return (None = all 40)
        seed: Random seed for sampling (only used if n_problems specified)

    Returns:
        List of problem dictionaries with keys:
        - problem: problem statement
        - answer: ground truth answer
        - unique_id: unique problem ID
        - subject: "AMC"
        - level: 3 (competition level)
    """
    # Try to load from local cache first
    local_file = DATA_DIR / "amc23.json"

    if local_file.exists():
        logger.info(f"Loading AMC23 from local cache: {local_file.name}")
        with open(local_file, 'r') as f:
            problems = json.load(f)
        logger.info(f"Loaded {len(problems)} AMC23 problems from cache")
    else:
        logger.info("Loading AMC23 dataset from HuggingFace...")
        dataset = load_dataset("math-ai/amc23", split="test")

        problems = []
        for idx, item in enumerate(dataset):
            problems.append({
                'problem': item['question'],
                'answer': str(item['answer']),  # Convert to string for consistency
                'unique_id': f"amc23_{idx}",
                'subject': 'AMC',
                'level': 3  # Competition level
            })

        logger.info(f"Loaded {len(problems)} AMC23 problems")

    # Sample if requested
    if n_problems is not None and n_problems < len(problems):
        import random
        random.seed(seed)
        problems = random.sample(problems, n_problems)
        logger.info(f"Sampled {len(problems)} problems (seed={seed})")

    return problems


def load_aime(
    n_problems: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load AIME dataset from local cache or HuggingFace.

    Dataset: gneubig/aime-1983-2024
    Size: 933 problems from 1983-2024 AIME competitions
    Levels: All competition level (assigned level 4 for consistency)

    Args:
        n_problems: Number of problems to return (None = all 933)
        seed: Random seed for sampling (only used if n_problems specified)

    Returns:
        List of problem dictionaries with keys:
        - problem: problem statement
        - answer: ground truth answer (0-999 integer)
        - unique_id: unique problem ID
        - subject: "AIME"
        - level: 4 (very hard competition level)
    """
    # Try to load from local cache first
    local_file = DATA_DIR / "aime.json"

    if local_file.exists():
        logger.info(f"Loading AIME from local cache: {local_file.name}")
        with open(local_file, 'r') as f:
            problems = json.load(f)
        logger.info(f"Loaded {len(problems)} AIME problems from cache")
    else:
        logger.info("Loading AIME dataset from HuggingFace...")
        dataset = load_dataset("gneubig/aime-1983-2024", split="train")

        problems = []
        for idx, item in enumerate(dataset):
            # Handle the one edge case where answer has two values
            answer = str(item['Answer']).strip()
            if 'or' in answer.lower():
                # Take first value for "080 or 081" case
                answer = answer.split()[0]

            problems.append({
                'problem': item['Question'],
                'answer': answer,
                'unique_id': item['ID'],
                'subject': 'AIME',
                'level': 4  # Very hard competition level
            })

        logger.info(f"Loaded {len(problems)} AIME problems")

    # Sample if requested
    if n_problems is not None and n_problems < len(problems):
        import random
        random.seed(seed)
        problems = random.sample(problems, n_problems)
        logger.info(f"Sampled {len(problems)} problems (seed={seed})")

    return problems


def load_gpqa(
    n_problems: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load GPQA dataset from local cache or HuggingFace.

    Dataset: Wanfq/gpqa (gpqa_main config)
    Size: Varies (only train split available, used for both train/test)
    Levels: All graduate-level science (assigned level 5 for consistency)

    Multiple choice format with 4 options (A-D) that are shuffled per question.
    The MC instruction is embedded in the question text.

    Args:
        n_problems: Number of problems to return (None = all)
        seed: Random seed for sampling and choice shuffling

    Returns:
        List of problem dictionaries with keys:
        - problem: problem statement with MC options and instruction
        - answer: correct choice letter (A, B, C, or D)
        - unique_id: unique problem ID
        - subject: "Science" (GPQA is graduate-level science)
        - level: 5 (graduate level)
    """
    # Try to load from local cache first
    local_file = DATA_DIR / "gpqa.json"

    if local_file.exists():
        logger.info(f"Loading GPQA from local cache: {local_file.name}")
        with open(local_file, 'r') as f:
            problems = json.load(f)
        logger.info(f"Loaded {len(problems)} GPQA problems from cache")
    else:
        logger.info("Loading GPQA dataset from HuggingFace...")
        dataset = load_dataset("Wanfq/gpqa", name="gpqa_main", split="train")

        choice_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        rng = random.Random(seed)

        problems = []
        for idx, item in enumerate(dataset):
            # Create list of all choices (correct + 3 incorrect)
            choices = [
                item['Correct Answer'],
                item['Incorrect Answer 1'],
                item['Incorrect Answer 2'],
                item['Incorrect Answer 3']
            ]

            # Shuffle the choices while tracking correct answer position
            choice_order = list(range(4))
            rng.shuffle(choice_order)
            shuffled_choices = [choices[i] for i in choice_order]

            # Find where the correct answer ended up (it was originally at index 0)
            correct_position = choice_order.index(0)
            correct_letter = choice_map[correct_position]

            # Format question with MC options and instruction
            formatted_question = (
                f"{item['Question']}\n"
                f" A) {shuffled_choices[0]}\n"
                f" B) {shuffled_choices[1]}\n"
                f" C) {shuffled_choices[2]}\n"
                f" D) {shuffled_choices[3]}\n\n"
                f"Your final answer should be a single choice letter in the form \\boxed{{answer}}, at the end of your response."
            )

            problems.append({
                'problem': formatted_question,
                'answer': correct_letter,
                'unique_id': f"gpqa_{idx}",
                'subject': 'Science',
                'level': 5  # Graduate level
            })

        logger.info(f"Loaded {len(problems)} GPQA problems")

    # Sample if requested
    if n_problems is not None and n_problems < len(problems):
        random.seed(seed)
        problems = random.sample(problems, n_problems)
        logger.info(f"Sampled {len(problems)} problems (seed={seed})")

    return problems


def load_csqa(
    n_problems: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load CommonsenseQA dataset from local cache or HuggingFace.

    Dataset: tau/commonsense_qa
    Size: Train and validation splits available
    Levels: All commonsense reasoning (assigned level 2 for consistency)

    Multiple choice format with 5 options (A-E).
    The MC instruction is embedded in the question text.

    Args:
        n_problems: Number of problems to return (None = all)
        seed: Random seed for sampling

    Returns:
        List of problem dictionaries with keys:
        - problem: problem statement with MC options and instruction
        - answer: correct choice letter (A, B, C, D, or E)
        - unique_id: unique problem ID
        - subject: "Commonsense"
        - level: 2 (commonsense level)
    """
    # Try to load from local cache first
    local_file = DATA_DIR / "csqa.json"

    if local_file.exists():
        logger.info(f"Loading CSQA from local cache: {local_file.name}")
        with open(local_file, 'r') as f:
            problems = json.load(f)
        logger.info(f"Loaded {len(problems)} CSQA problems from cache")
    else:
        logger.info("Loading CSQA dataset from HuggingFace...")
        dataset = load_dataset("tau/commonsense_qa")

        problems = []

        # Use validation split for testing (train is too large)
        for idx, item in enumerate(dataset['validation']):
            # Format question with MC options and instruction
            formatted_question = (
                f"{item['question']}\n"
                f" A) {item['choices']['text'][0]}\n"
                f" B) {item['choices']['text'][1]}\n"
                f" C) {item['choices']['text'][2]}\n"
                f" D) {item['choices']['text'][3]}\n"
                f" E) {item['choices']['text'][4]}\n\n"
                f"Your final answer should be a single choice letter in the form \\boxed{{answer}}, at the end of your response."
            )

            problems.append({
                'problem': formatted_question,
                'answer': item['answerKey'],
                'unique_id': f"csqa_{idx}",
                'subject': 'Commonsense',
                'level': 2  # Commonsense reasoning level
            })

        logger.info(f"Loaded {len(problems)} CSQA problems")

    # Sample if requested
    if n_problems is not None and n_problems < len(problems):
        random.seed(seed)
        problems = random.sample(problems, n_problems)
        logger.info(f"Sampled {len(problems)} problems (seed={seed})")

    return problems


def load_mathqa(
    n_problems: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load MathQA dataset from local cache or HuggingFace.

    Dataset: allenai/math_qa
    Size: Train and test splits available
    Levels: All math word problems (assigned level 2 for consistency)

    Multiple choice format with options already formatted in the dataset.
    The MC instruction is embedded in the question text.

    Args:
        n_problems: Number of problems to return (None = all)
        seed: Random seed for sampling

    Returns:
        List of problem dictionaries with keys:
        - problem: problem statement with MC options and instruction
        - answer: correct choice letter
        - unique_id: unique problem ID
        - subject: "Math"
        - level: 2 (word problem level)
    """
    # Try to load from local cache first
    local_file = DATA_DIR / "mathqa.json"

    if local_file.exists():
        logger.info(f"Loading MathQA from local cache: {local_file.name}")
        with open(local_file, 'r') as f:
            problems = json.load(f)
        logger.info(f"Loaded {len(problems)} MathQA problems from cache")
    else:
        logger.info("Loading MathQA dataset from HuggingFace...")
        dataset = load_dataset("allenai/math_qa")

        problems = []

        # Use test split
        for idx, item in enumerate(dataset['test']):
            # Format question with MC options and instruction
            # The 'options' field already contains formatted choices
            formatted_question = (
                f"{item['Problem']}\n"
                f"{item['options']}\n\n"
                f"Your final answer should be a single choice letter in the form \\boxed{{answer}}, at the end of your response."
            )

            problems.append({
                'problem': formatted_question,
                'answer': item['correct'],
                'unique_id': f"mathqa_{idx}",
                'subject': 'Math',
                'level': 2  # Math word problem level
            })

        logger.info(f"Loaded {len(problems)} MathQA problems")

    # Sample if requested
    if n_problems is not None and n_problems < len(problems):
        random.seed(seed)
        problems = random.sample(problems, n_problems)
        logger.info(f"Sampled {len(problems)} problems (seed={seed})")

    return problems


def load_svamp(
    n_problems: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load SVAMP dataset from local cache or HuggingFace.

    Dataset: ChilleD/SVAMP
    Size: 700 train problems, 300 test problems
    Levels: All grade school level (assigned level 2 for consistency)

    NOTE: SVAMP is NOT multiple choice - it requires numeric answers.

    Args:
        n_problems: Number of problems to return (None = all)
        seed: Random seed for sampling

    Returns:
        List of problem dictionaries with keys:
        - problem: problem statement
        - answer: numeric answer as string
        - unique_id: unique problem ID
        - subject: "Math"
        - level: 2 (grade school level)
    """
    # Try to load from local cache first
    local_file = DATA_DIR / "svamp.json"

    if local_file.exists():
        logger.info(f"Loading SVAMP from local cache: {local_file.name}")
        with open(local_file, 'r') as f:
            problems = json.load(f)
        logger.info(f"Loaded {len(problems)} SVAMP problems from cache")
    else:
        logger.info("Loading SVAMP dataset from HuggingFace...")
        dataset = load_dataset("ChilleD/SVAMP")

        problems = []

        # Use test split (300 problems)
        for idx, item in enumerate(dataset['test']):
            problems.append({
                'problem': item['question_concat'],
                'answer': str(item['Answer']),
                'unique_id': f"svamp_{idx}",
                'subject': 'Math',
                'level': 2  # Grade school level
            })

        logger.info(f"Loaded {len(problems)} SVAMP problems")

    # Sample if requested
    if n_problems is not None and n_problems < len(problems):
        random.seed(seed)
        problems = random.sample(problems, n_problems)
        logger.info(f"Sampled {len(problems)} problems (seed={seed})")

    return problems


# IMO-Bench CSV URL
IMOBENCH_CSV_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench.csv"


def load_imobench(
    n_problems: Optional[int] = None,
    imo_shortlist_only: bool = False,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load IMO-Bench dataset from local cache or GitHub.

    Dataset: google-deepmind/superhuman imobench/answerbench.csv
    Size: ~270 problems total, ~80 IMO Shortlist problems
    Levels: All IMO-level (assigned level 5 for consistency)

    Args:
        n_problems: Number of problems to return (None = all)
        imo_shortlist_only: If True, only return problems where Source contains "IMO Shortlist"
        seed: Random seed for sampling (only used if n_problems specified)

    Returns:
        List of problem dictionaries with keys:
        - problem: problem statement (LaTeX)
        - answer: ground truth answer (as-is from CSV)
        - unique_id: unique problem ID
        - subject: subject area (Algebra/Combinatorics/Geometry)
        - level: 5 (IMO level)
    """
    # Try to load from local cache first
    if imo_shortlist_only:
        local_file = DATA_DIR / "imo.json"
        dataset_name = "IMO Shortlist"
    else:
        local_file = DATA_DIR / "imobench.json"
        dataset_name = "IMO-Bench"

    if local_file.exists():
        logger.info(f"Loading {dataset_name} from local cache: {local_file.name}")
        with open(local_file, 'r') as f:
            problems = json.load(f)
        logger.info(f"Loaded {len(problems)} {dataset_name} problems from cache")
    else:
        logger.info(f"Loading {dataset_name} from GitHub...")
        try:
            with urlopen(IMOBENCH_CSV_URL) as response:
                content = response.read().decode('utf-8')
        except URLError as e:
            raise RuntimeError(f"Failed to fetch IMO-Bench CSV from GitHub: {e}")

        reader = csv.DictReader(io.StringIO(content))

        problems = []
        for row in reader:
            # Filter by source if imo_shortlist_only
            if imo_shortlist_only and "IMO Shortlist" not in row.get('Source', ''):
                continue

            problems.append({
                'problem': row['Problem'],
                'answer': row['Short Answer'],
                'unique_id': row['Problem ID'],
                'subject': row['Category'],
                'level': 5  # IMO level - hardest
            })

        logger.info(f"Loaded {len(problems)} {dataset_name} problems from GitHub")

    # Sample if requested
    if n_problems is not None and n_problems < len(problems):
        random.seed(seed)
        problems = random.sample(problems, n_problems)
        logger.info(f"Sampled {len(problems)} problems (seed={seed})")

    return problems


def load_imo(
    n_problems: Optional[int] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load IMO Shortlist problems only (subset of IMO-Bench).

    This is a convenience wrapper around load_imobench() that filters
    to only include problems where Source contains "IMO Shortlist".

    Dataset: google-deepmind/superhuman imobench/answerbench.csv (filtered)
    Size: ~80 problems
    Levels: All IMO-level (assigned level 5 for consistency)

    Args:
        n_problems: Number of problems to return (None = all ~80)
        seed: Random seed for sampling (only used if n_problems specified)

    Returns:
        List of problem dictionaries with keys:
        - problem: problem statement (LaTeX)
        - answer: ground truth answer (as-is from CSV)
        - unique_id: unique problem ID
        - subject: subject area (Algebra/Combinatorics/Geometry)
        - level: 5 (IMO level)
    """
    return load_imobench(n_problems=n_problems, imo_shortlist_only=True, seed=seed)


def load_dataset_by_name(
    dataset_name: str,
    n_problems: Optional[int] = None,
    level: Optional[int] = None,
    split: str = "test",
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Load a dataset by name with unified interface.

    Args:
        dataset_name: Name of dataset ("math500", "gsm8k", "amc23", or "aime")
        n_problems: Number of problems to return (None = all)
        level: For MATH-500, filter by difficulty level (1-5)
        split: For GSM8K, dataset split ("test" or "train")
        seed: Random seed for sampling

    Returns:
        List of problem dictionaries with standardized fields

    Raises:
        ValueError: If dataset_name is not recognized
    """
    dataset_name = dataset_name.lower()

    if dataset_name in ['math500', 'math-500', 'math_500']:
        return load_math500(n_problems=n_problems, level=level, seed=seed)

    elif dataset_name in ['gsm8k', 'gsm-8k', 'gsm_8k']:
        return load_gsm8k(n_problems=n_problems, split=split, seed=seed)

    elif dataset_name in ['amc23', 'amc-23', 'amc_23', 'amc']:
        return load_amc23(n_problems=n_problems, seed=seed)

    elif dataset_name in ['aime', 'aime-1983-2024']:
        return load_aime(n_problems=n_problems, seed=seed)

    elif dataset_name in ['gpqa', 'gpqa_main']:
        return load_gpqa(n_problems=n_problems, seed=seed)

    elif dataset_name in ['csqa', 'commonsense_qa', 'commonsenseqa']:
        return load_csqa(n_problems=n_problems, seed=seed)

    elif dataset_name in ['mathqa', 'math_qa']:
        return load_mathqa(n_problems=n_problems, seed=seed)

    elif dataset_name in ['svamp']:
        return load_svamp(n_problems=n_problems, seed=seed)

    elif dataset_name in ['imo', 'imo_shortlist', 'imo-shortlist']:
        return load_imo(n_problems=n_problems, seed=seed)

    elif dataset_name in ['imobench', 'imo-bench', 'imo_bench']:
        return load_imobench(n_problems=n_problems, imo_shortlist_only=False, seed=seed)

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: math500, gsm8k, amc23, aime, gpqa, csqa, mathqa, svamp, imo, imobench"
        )


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about a dataset.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary with dataset metadata
    """
    dataset_name = dataset_name.lower()

    info = {
        'math500': {
            'name': 'MATH-500',
            'huggingface_id': 'HuggingFaceH4/MATH-500',
            'size': 500,
            'splits': ['test'],
            'levels': [1, 2, 3, 4, 5],
            'subjects': ['Algebra', 'Number Theory', 'Counting & Probability',
                        'Geometry', 'Intermediate Algebra', 'Precalculus', 'Prealgebra'],
            'description': '500 challenging math competition problems from MATH dataset'
        },
        'gsm8k': {
            'name': 'GSM8K',
            'huggingface_id': 'gsm8k',
            'size': {'train': 7473, 'test': 1319},
            'splits': ['train', 'test'],
            'levels': [2],  # Grade school level
            'subjects': ['Math'],
            'description': 'Grade school math word problems requiring multi-step reasoning'
        },
        'amc23': {
            'name': 'AMC23',
            'huggingface_id': 'math-ai/amc23',
            'size': 40,
            'splits': ['test'],
            'levels': [3],  # Competition level
            'subjects': ['AMC'],
            'description': '40 problems from 2023 AMC math competition'
        },
        'aime': {
            'name': 'AIME',
            'huggingface_id': 'gneubig/aime-1983-2024',
            'size': 933,
            'splits': ['train'],
            'levels': [4],  # Very hard competition level
            'subjects': ['AIME'],
            'description': '933 problems from 1983-2024 AIME competitions (answers 0-999)'
        },
        'gpqa': {
            'name': 'GPQA',
            'huggingface_id': 'Wanfq/gpqa',
            'size': 'varies',
            'splits': ['train'],
            'levels': [5],  # Graduate level
            'subjects': ['Science'],
            'description': 'Graduate-level science multiple choice questions (4 options, A-D)',
            'multiple_choice': True
        },
        'csqa': {
            'name': 'CommonsenseQA',
            'huggingface_id': 'tau/commonsense_qa',
            'size': {'train': 9741, 'validation': 1221},
            'splits': ['train', 'validation'],
            'levels': [2],  # Commonsense level
            'subjects': ['Commonsense'],
            'description': 'Commonsense reasoning multiple choice questions (5 options, A-E)',
            'multiple_choice': True
        },
        'mathqa': {
            'name': 'MathQA',
            'huggingface_id': 'allenai/math_qa',
            'size': {'train': 29837, 'test': 2985, 'validation': 4475},
            'splits': ['train', 'test', 'validation'],
            'levels': [2],  # Math word problem level
            'subjects': ['Math'],
            'description': 'Math word problems with multiple choice answers',
            'multiple_choice': True
        },
        'svamp': {
            'name': 'SVAMP',
            'huggingface_id': 'ChilleD/SVAMP',
            'size': {'train': 700, 'test': 300},
            'splits': ['train', 'test'],
            'levels': [2],  # Grade school level
            'subjects': ['Math'],
            'description': 'Grade school math word problems (numeric answers)',
            'multiple_choice': False
        },
        'imo': {
            'name': 'IMO Shortlist',
            'source': 'google-deepmind/superhuman (filtered)',
            'url': 'https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench.csv',
            'size': 123,  # filtered to IMO Shortlist only
            'splits': ['all'],
            'levels': [5],  # IMO level - hardest
            'subjects': ['Algebra', 'Combinatorics', 'Number theory'],
            'description': 'IMO Shortlist problems only (subset of IMO-Bench)',
            'multiple_choice': False
        },
        'imobench': {
            'name': 'IMO-Bench',
            'source': 'google-deepmind/superhuman',
            'url': 'https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench.csv',
            'size': 400,
            'splits': ['all'],
            'levels': [5],  # IMO level - hardest
            'subjects': ['Algebra', 'Combinatorics', 'Geometry', 'Number theory'],
            'description': 'Full IMO-Bench AnswerBench dataset from DeepMind (400 olympiad problems)',
            'multiple_choice': False
        }
    }

    # Normalize dataset name
    if dataset_name in ['math500', 'math-500', 'math_500']:
        return info['math500']
    elif dataset_name in ['gsm8k', 'gsm-8k', 'gsm_8k']:
        return info['gsm8k']
    elif dataset_name in ['amc23', 'amc-23', 'amc_23', 'amc']:
        return info['amc23']
    elif dataset_name in ['aime', 'aime-1983-2024']:
        return info['aime']
    elif dataset_name in ['gpqa', 'gpqa_main']:
        return info['gpqa']
    elif dataset_name in ['csqa', 'commonsense_qa', 'commonsenseqa']:
        return info['csqa']
    elif dataset_name in ['mathqa', 'math_qa']:
        return info['mathqa']
    elif dataset_name in ['svamp']:
        return info['svamp']
    elif dataset_name in ['imo', 'imo_shortlist', 'imo-shortlist']:
        return info['imo']
    elif dataset_name in ['imobench', 'imo-bench', 'imo_bench']:
        return info['imobench']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    """Test data loaders."""

    # Test MATH-500
    print("\n" + "="*80)
    print("Testing MATH-500 Loader")
    print("="*80)
    math500_problems = load_math500(n_problems=5, level=5, seed=42)
    print(f"Loaded {len(math500_problems)} MATH-500 problems")
    if math500_problems:
        print(f"Example problem:")
        print(f"  Problem: {math500_problems[0]['problem'][:100]}...")
        print(f"  Answer: {math500_problems[0]['answer']}")
        print(f"  Subject: {math500_problems[0]['subject']}")
        print(f"  Level: {math500_problems[0]['level']}")

    # Test GSM8K
    print("\n" + "="*80)
    print("Testing GSM8K Loader")
    print("="*80)
    gsm8k_problems = load_gsm8k(n_problems=5, seed=42)
    print(f"Loaded {len(gsm8k_problems)} GSM8K problems")
    if gsm8k_problems:
        print(f"Example problem:")
        print(f"  Problem: {gsm8k_problems[0]['problem'][:100]}...")
        print(f"  Answer: {gsm8k_problems[0]['answer']}")
        print(f"  Subject: {gsm8k_problems[0]['subject']}")
        print(f"  Level: {gsm8k_problems[0]['level']}")

    # Test AMC23
    print("\n" + "="*80)
    print("Testing AMC23 Loader")
    print("="*80)
    amc23_problems = load_amc23(n_problems=5, seed=42)
    print(f"Loaded {len(amc23_problems)} AMC23 problems")
    if amc23_problems:
        print(f"Example problem:")
        print(f"  Problem: {amc23_problems[0]['problem'][:100]}...")
        print(f"  Answer: {amc23_problems[0]['answer']}")
        print(f"  Subject: {amc23_problems[0]['subject']}")
        print(f"  Level: {amc23_problems[0]['level']}")

    # Test unified interface
    print("\n" + "="*80)
    print("Testing Unified Interface")
    print("="*80)
    for dataset_name in ['math500', 'gsm8k', 'amc23']:
        print(f"\nDataset: {dataset_name}")
        info = get_dataset_info(dataset_name)
        print(f"  Name: {info['name']}")
        print(f"  HuggingFace ID: {info['huggingface_id']}")
        print(f"  Size: {info['size']}")
        print(f"  Description: {info['description']}")
