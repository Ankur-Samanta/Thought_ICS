#!/usr/bin/env python3
"""
Merge split oracle resample result files.

Usage:
  python merge_oracle_resample_splits.py gptoss120b
  python merge_oracle_resample_splits.py gptoss120b --dry-run
"""

import json
import argparse
from pathlib import Path

RESULTS_DIR = Path("paper/results")

# Split configurations: model -> {exp_type: [(start, end), ...]}
SPLIT_CONFIG = {
    'gptoss120b': {
        'thought': [(0, 222), (222, 444), (444, 666), (666, 888)],
        'token': [(0, 228), (228, 456), (456, 683), (683, 910)],
    }
}


def merge_splits(model: str, dry_run: bool = False):
    """Merge split files for a model."""
    if model not in SPLIT_CONFIG:
        print(f"No split config for {model}")
        return

    config = SPLIT_CONFIG[model]

    for exp_type, splits in config.items():
        part_files = []
        all_exist = True

        for start, end in splits:
            part_file = RESULTS_DIR / f"oracle_resample_{exp_type}_{model}_part{start}-{end}.json"
            part_files.append(part_file)
            if not part_file.exists():
                print(f"  Missing: {part_file}")
                all_exist = False

        if not all_exist:
            print(f"Skipping {exp_type} - missing part files")
            continue

        # Merge
        merged = []
        for part_file in part_files:
            data = json.load(open(part_file))
            print(f"  {part_file.name}: {len(data)} samples")
            merged.extend(data)

        output_file = RESULTS_DIR / f"oracle_resample_{exp_type}_{model}.json"
        print(f"  -> {output_file.name}: {len(merged)} samples total")

        if not dry_run:
            with open(output_file, 'w') as f:
                json.dump(merged, f, indent=2)
            print(f"  Saved!")
        else:
            print(f"  (dry-run, not saved)")

        print()


def main():
    parser = argparse.ArgumentParser(description='Merge split oracle resample files')
    parser.add_argument('model', type=str, help='Model to merge (e.g., gptoss120b)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be merged without saving')
    args = parser.parse_args()

    print(f"Merging splits for {args.model}...")
    print()
    merge_splits(args.model, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
