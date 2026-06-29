#!/usr/bin/env python3
"""
Simply merge three JSONL files
"""

import json
from pathlib import Path

def simple_merge():
    """Simply merge three JSONL files"""

    # File paths
    base_dir = Path("maindata")
    files = [
        base_dir / "processed_train_data.jsonl",
        base_dir / "processed_test_data.jsonl",
        base_dir / "processed_eval_data.jsonl"
    ]
    output_file = base_dir / "merged_data.jsonl"

    print("Starting to merge JSONL files...")

    total_count = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in files:
            print(f"Processing {file_path.name}...")
            count = 0

            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if line.strip():  # Skip empty lines
                        outfile.write(line)
                        count += 1

            print(f"  - Added {count} records")
            total_count += count

    print(f"\nMerge complete!")
    print(f"Output file: {output_file}")
    print(f"Total records: {total_count}")

if __name__ == "__main__":
    simple_merge()


