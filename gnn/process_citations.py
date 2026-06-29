#!/usr/bin/env python3
"""
Split data files by percentile based on the n_citation field
- 1st-10th percentile: saved to low_cited/
- 90th-99th percentile: saved to high_cited/
"""

import json
import os
import numpy as np
from pathlib import Path

def process_jsonl_file(input_file, output_low_file, output_high_file):
    """Process a single JSONL file, splitting by n_citation percentile"""
    print(f"Processing file: {input_file}")

    # Step 1: read all data and extract n_citation values
    all_data = []
    all_citations = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse the JSON line
                data = json.loads(line)

                # Get the n_citation value
                n_citation = None
                if isinstance(data, dict):
                    # If it is a dict, look for the n_citation field
                    if 'n_citation' in data:
                        n_citation = data['n_citation']
                    else:
                        # If the data is nested, try to get it from the values
                        for key, value in data.items():
                            if isinstance(value, dict) and 'n_citation' in value:
                                n_citation = value['n_citation']
                                break

                if n_citation is not None:
                    all_data.append((line, n_citation))
                    all_citations.append(n_citation)
                else:
                    print(f"Warning: no n_citation field found on line {line_num}")

            except json.JSONDecodeError as e:
                print(f"JSON parse error (line {line_num}): {e}")
                continue

    if not all_citations:
        print(f"  Error: no valid n_citation data found")
        return

    # Step 2: compute percentile thresholds
    all_citations = np.array(all_citations)
    low_threshold_10 = np.percentile(all_citations, 10)  # 10th percentile
    high_threshold_90 = np.percentile(all_citations, 90)  # 90th percentile

    print(f"  Citation statistics:")
    print(f"    Min: {all_citations.min()}")
    print(f"    Max: {all_citations.max()}")
    print(f"    Median: {np.median(all_citations):.2f}")
    print(f"    10th percentile: {low_threshold_10:.2f}")
    print(f"    90th percentile: {high_threshold_90:.2f}")

    # Step 3: classify by percentile and write to files
    low_count = 0
    high_count = 0
    other_count = 0

    with open(output_low_file, 'w', encoding='utf-8') as lowfile, \
         open(output_high_file, 'w', encoding='utf-8') as highfile:

        for line, n_citation in all_data:
            if n_citation <= low_threshold_10:
                # 1st-10th percentile
                lowfile.write(line + '\n')
                low_count += 1
            elif n_citation >= high_threshold_90:
                # 90th-99th percentile
                highfile.write(line + '\n')
                high_count += 1
            else:
                # Other percentiles (not saved)
                other_count += 1

    print(f"  Total: {len(all_data)} records")
    print(f"  Low-cited (1st-10th percentile): {low_count} records")
    print(f"  High-cited (90th-99th percentile): {high_count} records")
    print(f"  Other (11th-89th percentile): {other_count} records")
    print()

def main():
    # Base path
    base_path = Path("maindata")
    high_cited_path = base_path / "high_cited"
    low_cited_path = base_path / "low_cited"

    # List of source folders
    source_folders = [
        "Openalex/test1_distribution_aware",
        "Openalex/test2_distribution_aware", 
        "Openalex/test3_distribution_aware",
        "test1",
        "test2",
        "test3"
    ]
    
    # Filename patterns to process
    processed_files = [
        "train_processsed.jsonl",
        "eval_processsed.jsonl",
        "test_processsed.jsonl"
    ]

    print("Starting to process files by percentile...")
    print("=" * 50)

    for folder in source_folders:
        folder_path = base_path / folder
        if not folder_path.exists():
            print(f"Folder does not exist: {folder_path}")
            continue

        print(f"Processing folder: {folder}")

        # Process the processed files in the main folder
        for filename in processed_files:
            input_file = folder_path / filename
            if input_file.exists():
                # Generate the output filename
                output_name = f"{folder.replace('/', '_')}_{filename}"
                output_low = low_cited_path / output_name
                output_high = high_cited_path / output_name

                process_jsonl_file(input_file, output_low, output_high)

        # Process the qwen3 subfolder inside the test1 folder
        if folder == "test1":
            qwen3_path = folder_path / "qwen3"
            if qwen3_path.exists():
                print(f"Processing subfolder: {folder}/qwen3")
                for filename in processed_files:
                    input_file = qwen3_path / filename
                    if input_file.exists():
                        output_name = f"{folder}_qwen3_{filename}"
                        output_low = low_cited_path / output_name
                        output_high = high_cited_path / output_name

                        process_jsonl_file(input_file, output_low, output_high)

    print("=" * 50)
    print("Percentile-based processing complete!")

    # Show result statistics
    print("\nResult statistics:")
    print(f"High-cited files (90th-99th percentile) saved in: {high_cited_path}")
    print(f"Low-cited files (1st-10th percentile) saved in: {low_cited_path}")

    high_files = list(high_cited_path.glob("*.jsonl"))
    low_files = list(low_cited_path.glob("*.jsonl"))

    print(f"Generated {len(high_files)} high-cited files")
    print(f"Generated {len(low_files)} low-cited files")

if __name__ == "__main__":
    main()
