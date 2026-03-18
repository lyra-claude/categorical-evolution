#!/usr/bin/env python3
"""
Restore checkers CSV by merging old data (from git) with truncated current data.

The current CSV was corrupted: multiprocessing write lost the header.
This script:
1. Extracts old data from git commit 7bc130d
2. Reads the (corrupted) current CSV
3. Merges them, deduplicating by (topology, seed, generation)
4. Writes merged result with proper header
5. Reports coverage

Usage:
    python3 restore_checkers.py                    # Dry run (print what would happen)
    python3 restore_checkers.py --execute          # Actually merge and write
"""

import subprocess
import pandas as pd
import sys
from pathlib import Path
from io import StringIO
from collections import Counter

GIT_COMMIT = "7bc130d"
CSV_PATH = Path("experiments/experiment_e_checkers.csv")
HEADER = "topology,seed,generation,hamming_diversity,population_divergence,best_fitness,island_0_diversity,island_0_fitness,island_1_diversity,island_1_fitness,island_2_diversity,island_2_fitness,island_3_diversity,island_3_fitness,island_4_diversity,island_4_fitness,div_0_1,div_0_2,div_0_3,div_0_4,div_1_2,div_1_3,div_1_4,div_2_3,div_2_4,div_3_4"


def extract_old_data():
    """Extract CSV from git commit."""
    print(f"[*] Extracting old data from git commit {GIT_COMMIT}...")
    try:
        result = subprocess.run(
            ["git", "show", f"{GIT_COMMIT}:experiments/experiment_e_checkers.csv"],
            capture_output=True,
            text=True,
            check=True,
        )
        old_csv_text = result.stdout
        print(f"    ✓ Extracted {len(old_csv_text.splitlines())} lines")
        return old_csv_text
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Failed to extract from git: {e.stderr}")
        sys.exit(1)


def read_current_csv():
    """Read current (corrupted) CSV. Return raw lines."""
    print(f"[*] Reading current CSV from {CSV_PATH}...")
    try:
        with open(CSV_PATH, 'r') as f:
            lines = f.readlines()
        print(f"    ✓ Read {len(lines)} lines")
        return lines
    except Exception as e:
        print(f"    ✗ Failed to read: {e}")
        sys.exit(1)


def parse_csv_text(text, label, is_current=False):
    """Parse CSV text into DataFrame. Skip bad rows.

    For current CSV, handle carriage returns and corrupted headers.
    """
    lines = text.strip().split('\n')
    if not lines:
        return pd.DataFrame()

    # Clean carriage returns
    lines = [line.replace('\r', '') for line in lines]

    # For current data, skip the corrupted first line if it doesn't have proper header
    start_idx = 0
    if is_current and len(lines) > 0:
        # Check if first line looks like a proper header
        first_line_parts = lines[0].split(',')
        # Header should have 27 columns
        if len(first_line_parts) != 27:
            # First line is data, not header
            start_idx = 1
            header_from_spec = HEADER.split(',')
        else:
            # Check if first line has proper header names
            if 'topology' not in lines[0]:
                start_idx = 1
                header_from_spec = HEADER.split(',')
            else:
                header_from_spec = first_line_parts

    if start_idx == 0 and not is_current:
        header = lines[0].split(',')
        data_lines = lines[1:]
    elif start_idx > 0:
        header = header_from_spec
        data_lines = lines[start_idx:]
    else:
        header = lines[0].split(',')
        data_lines = lines[1:]

    data = []
    skipped = 0

    for i, line in enumerate(data_lines):
        if not line.strip():  # Skip empty lines
            continue
        parts = line.split(',')
        if len(parts) != len(header):
            skipped += 1
            continue
        data.append(parts)

    if skipped > 0:
        print(f"    ⚠ Skipped {skipped} malformed rows")

    if not data:
        print(f"    ⚠ No valid data rows found ({label})")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=header)

    # Convert dtypes
    df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
    df['generation'] = pd.to_numeric(df['generation'], errors='coerce')

    # Drop any rows with NaN in key columns
    df = df.dropna(subset=['topology', 'seed', 'generation'])
    df['seed'] = df['seed'].astype(int)
    df['generation'] = df['generation'].astype(int)

    print(f"    ✓ Parsed {len(df)} valid rows ({label})")
    return df


def merge_data(old_df, current_df):
    """Merge old and current, deduplicating by (topology, seed, generation)."""
    print("[*] Merging data...")

    # Combine
    merged = pd.concat([old_df, current_df], ignore_index=True)
    print(f"    Combined: {len(merged)} rows")

    # Deduplicate by (topology, seed, generation) — keep first occurrence
    merged_dedup = merged.drop_duplicates(
        subset=['topology', 'seed', 'generation'],
        keep='first'
    )
    print(f"    After dedup: {len(merged_dedup)} rows")
    print(f"    Removed {len(merged) - len(merged_dedup)} duplicates")

    return merged_dedup


def analyze_coverage(df):
    """Analyze (topology, seed) pair coverage and gaps."""
    print("[*] Analyzing coverage...")

    topologies = df['topology'].unique()
    print(f"    Topologies: {sorted(topologies)}")

    stats = []
    for topo in sorted(topologies):
        topo_df = df[df['topology'] == topo]
        seeds = sorted(topo_df['seed'].unique())
        gen_counts = topo_df.groupby('seed')['generation'].apply(lambda x: len(x)).to_dict()

        num_seeds = len(seeds)
        seed_range = f"{min(seeds)}-{max(seeds)}" if seeds else "N/A"
        avg_gens = sum(gen_counts.values()) / len(gen_counts) if gen_counts else 0

        # Identify gaps
        expected_seeds = set(range(0, 30))
        actual_seeds = set(seeds)
        missing = expected_seeds - actual_seeds

        stats.append({
            'topology': topo,
            'num_seeds': num_seeds,
            'seed_range': seed_range,
            'avg_gens_per_seed': round(avg_gens, 1),
            'missing_seeds': sorted(list(missing)) if missing else None,
        })

        status = "✓" if num_seeds == 30 else "⚠"
        print(f"    {status} {topo:15} {num_seeds:2}/30 seeds, avg {avg_gens:5.1f} gen/seed")
        if missing:
            print(f"       Missing: {missing}")

    return stats


def main():
    import os

    # Check if --execute flag is present
    execute = "--execute" in sys.argv
    mode = "EXECUTE" if execute else "DRY RUN"

    print(f"\n{'='*60}")
    print(f"CHECKERS CSV RESTORATION ({mode})")
    print(f"{'='*60}\n")

    # Ensure we're in the right directory
    if not Path("experiments").exists():
        print("✗ Not in categorical-evolution root. Run from there.")
        sys.exit(1)

    # Extract old data from git
    old_csv_text = extract_old_data()

    # Read current CSV (raw lines for inspection)
    current_lines = read_current_csv()

    # Parse both
    old_df = parse_csv_text(old_csv_text, "old from git", is_current=False)

    # Current CSV is corrupted — try to parse as-is
    current_text = ''.join(current_lines)
    current_df = parse_csv_text(current_text, "current (corrupted)", is_current=True)

    # Merge
    merged_df = merge_data(old_df, current_df)

    # Restore header order
    header_cols = HEADER.split(',')
    merged_df = merged_df[header_cols]

    # Analyze coverage
    coverage = analyze_coverage(merged_df)

    # Summary
    print("\n[*] Summary:")
    print(f"    Old data:     {len(old_df):6} rows")
    print(f"    Current data: {len(current_df):6} rows")
    print(f"    After merge:  {len(merged_df):6} rows (deduplicated)")

    total_pairs = len(merged_df.groupby(['topology', 'seed']))
    print(f"    (topology, seed) pairs: {total_pairs}")

    # Check for complete coverage
    complete = all(c['num_seeds'] == 30 for c in coverage)
    if complete:
        print(f"    ✓ COMPLETE: All topologies have 30 seeds")
    else:
        incomplete = [c['topology'] for c in coverage if c['num_seeds'] < 30]
        print(f"    ⚠ INCOMPLETE: {', '.join(incomplete)}")

    # Show what would happen
    if not execute:
        print(f"\n[*] DRY RUN MODE: No changes made.")
        print(f"    To execute: python3 restore_checkers.py --execute")
        return

    # Actually write
    print(f"\n[*] EXECUTING merge...")

    # Write to temp file first
    temp_path = CSV_PATH.with_suffix('.csv.tmp')
    print(f"    Writing to temp: {temp_path}...")
    merged_df.to_csv(temp_path, index=False)

    # Verify temp file
    verify_df = pd.read_csv(temp_path)
    print(f"    ✓ Verified temp: {len(verify_df)} rows")

    # Replace original
    print(f"    Replacing {CSV_PATH}...")
    import shutil
    shutil.move(str(temp_path), str(CSV_PATH))

    print(f"\n✓ DONE. Merged CSV written to {CSV_PATH}")
    print(f"  Rows: {len(merged_df)}")
    print(f"  Columns: {len(header_cols)}")


if __name__ == '__main__':
    main()
