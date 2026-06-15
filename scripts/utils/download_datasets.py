#!/usr/bin/env python3
"""
Download all datasets locally for offline access.

This script downloads MATH-500, GSM8K, AIME, and AMC23 datasets
and saves them locally in the same format as the HuggingFace datasets.
"""

import json
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_math500():
    """Download MATH-500 dataset."""
    logger.info("Downloading MATH-500 dataset...")

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    problems = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing MATH-500")):
        problems.append({
            'problem': item['problem'],
            'answer': item['answer'],
            'unique_id': item.get('unique_id', f"math500_{idx}"),
            'subject': item['subject'],
            'level': item['level']
        })

    output_file = DATA_DIR / "math500.json"
    with open(output_file, 'w') as f:
        json.dump(problems, f, indent=2)

    logger.info(f"✓ Saved {len(problems)} MATH-500 problems to {output_file}")

    # Create level-specific files for faster filtering
    for level in range(1, 6):
        level_problems = [p for p in problems if p['level'] == level]
        level_file = DATA_DIR / f"math500_level{level}.json"
        with open(level_file, 'w') as f:
            json.dump(level_problems, f, indent=2)
        logger.info(f"✓ Saved {len(level_problems)} level {level} problems to {level_file}")


def download_gsm8k():
    """Download GSM8K dataset."""
    logger.info("Downloading GSM8K dataset...")

    for split in ["train", "test"]:
        dataset = load_dataset("gsm8k", "main", split=split)

        problems = []
        for idx, item in enumerate(tqdm(dataset, desc=f"Processing GSM8K {split}")):
            # Extract answer from "#### 42" format
            answer_text = item['answer']
            if '####' in answer_text:
                clean_answer = answer_text.split('####')[-1].strip().replace(',', '')
            else:
                clean_answer = answer_text

            problems.append({
                'problem': item['question'],
                'answer': clean_answer,
                'unique_id': f"gsm8k_{split}_{idx}",
                'subject': 'Math',
                'level': 2
            })

        output_file = DATA_DIR / f"gsm8k_{split}.json"
        with open(output_file, 'w') as f:
            json.dump(problems, f, indent=2)

        logger.info(f"✓ Saved {len(problems)} GSM8K {split} problems to {output_file}")


def download_aime():
    """Download AIME dataset."""
    logger.info("Downloading AIME dataset...")

    dataset = load_dataset("gneubig/aime-1983-2024", split="train")

    problems = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing AIME")):
        # Handle edge case where answer has two values
        answer = str(item['Answer']).strip()
        if 'or' in answer.lower():
            answer = answer.split()[0]

        problems.append({
            'problem': item['Question'],
            'answer': answer,
            'unique_id': item['ID'],
            'subject': 'AIME',
            'level': 4
        })

    output_file = DATA_DIR / "aime.json"
    with open(output_file, 'w') as f:
        json.dump(problems, f, indent=2)

    logger.info(f"✓ Saved {len(problems)} AIME problems to {output_file}")


def download_amc23():
    """Download AMC23 dataset."""
    logger.info("Downloading AMC23 dataset...")

    dataset = load_dataset("math-ai/amc23", split="test")

    problems = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing AMC23")):
        problems.append({
            'problem': item['question'],
            'answer': str(item['answer']),
            'unique_id': f"amc23_{idx}",
            'subject': 'AMC',
            'level': 3
        })

    output_file = DATA_DIR / "amc23.json"
    with open(output_file, 'w') as f:
        json.dump(problems, f, indent=2)

    logger.info(f"✓ Saved {len(problems)} AMC23 problems to {output_file}")


def main():
    """Download all datasets."""
    logger.info("="*80)
    logger.info("DOWNLOADING ALL DATASETS")
    logger.info("="*80)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info("")

    try:
        download_math500()
        logger.info("")

        download_gsm8k()
        logger.info("")

        download_aime()
        logger.info("")

        download_amc23()
        logger.info("")

        logger.info("="*80)
        logger.info("✓ ALL DATASETS DOWNLOADED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Data saved to: {DATA_DIR}")
        logger.info("")
        logger.info("Files created:")
        for file in sorted(DATA_DIR.glob("*.json")):
            size_mb = file.stat().st_size / 1024 / 1024
            logger.info(f"  - {file.name} ({size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"Error downloading datasets: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
